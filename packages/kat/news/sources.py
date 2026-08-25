"""Pluggable news providers, free ones only.

The hard constraint this module encodes: **the platform must never require a
paid API.** Every provider here reads a public endpoint that costs nothing and
needs no key -- RSS/Atom feeds from wire services and public broadcasters, and
open government JSON APIs (Federal Register, NWS/NOAA). A commercial provider
can be added later as one more :class:`NewsSource` subclass, but nothing
upstream is allowed to assume one exists.

Why two provider classes and not one
------------------------------------
RSS is a fixed schema, so :class:`RSSSource` needs no per-feed configuration
beyond a URL. Open JSON APIs are all different, so :class:`JSONSource` takes a
small path-mapping description instead of a bespoke subclass per endpoint --
adding NOAA alerts or Federal Register documents is a YAML entry, not code.
Anything genuinely irregular still gets a subclass; the mapping language is
deliberately not a programming language.

What providers do not do
------------------------
A provider fetches and extracts fields. It does not normalise text, strip HTML,
deduplicate, assign an article id, or read a clock. Those belong to
:mod:`kat.news.ingest`, which owns the ``ingested_ts`` stamp that the whole
point-in-time guarantee rests on. Keeping providers dumb is what makes them
safe to add.

Conditional requests
--------------------
Each source remembers the ``ETag`` and ``Last-Modified`` of its last successful
response and sends them back. A feed polled every five minutes changes far less
often than that, so most polls cost a 304 and no body. This is politeness that
also happens to be cheaper: several of these publishers rate-limit aggressively
and a client that re-downloads an unchanged 200 KB feed all day gets blocked.
"""

from __future__ import annotations

import asyncio
import calendar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from kat.common.errors import ConfigError, KatError
from kat.common.timeutil import parse_rfc3339
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DEFAULT_SOURCES_FILE",
    "JSONFieldMap",
    "JSONSource",
    "NewsSource",
    "NewsSourceError",
    "RSSSource",
    "SourceKind",
    "SourceSpec",
    "build_source",
    "load_source_specs",
    "load_sources",
]

#: Path is relative to the repository root; :func:`load_sources` resolves it.
DEFAULT_SOURCES_FILE = "configs/news_sources.yaml"

_REPO_ROOT = Path(__file__).resolve().parents[3]


class NewsSourceError(KatError):
    """A provider could not produce items from a response.

    Distinct from a transport failure: the ingestor treats a transport error as
    transient and a :class:`NewsSourceError` as a signal that the feed's shape
    has changed and the mapping needs attention.
    """


class SourceKind(str, Enum):
    RSS = "rss"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class JSONFieldMap:
    """How to pull :class:`~kat.news.types.RawItem` fields out of a JSON payload.

    Paths are dotted, with two extra segment forms: an integer indexes a list,
    and ``*`` maps the remainder of the path across a list and returns every
    match. ``properties.headline`` and ``agencies.*.name`` are both valid. This
    is intentionally not JSONPath -- three rules cover every free feed we use,
    and a full expression language in a config file is an injection surface
    without a corresponding benefit.
    """

    items_path: str = ""
    external_id: str = "id"
    title: str = "title"
    summary: str = ""
    body: str = ""
    url: str = ""
    published: str = ""
    published_format: str = "auto"
    categories: str = ""
    language: str = ""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Declarative description of one feed, as it appears in YAML.

    ``reliability`` is a prior in [0, 1] on how much weight a story from this
    source should carry before any evidence about the specific story. Primary
    government sources sit high because they *are* the settlement source for
    many Kalshi markets; aggregators sit low because they republish without
    verifying. :mod:`kat.news.impact` blends this prior with observed
    corroboration rather than trusting it outright.
    """

    source_id: str
    name: str
    url: str
    kind: SourceKind = SourceKind.RSS
    enabled: bool = True
    reliability: float = 0.5
    categories: tuple[str, ...] = ()
    # Politeness floor between two requests to this source, independent of the
    # global poll interval. Government APIs publish theirs; wires do not, so the
    # default is conservative.
    min_interval_s: float = 60.0
    timeout_s: float = 15.0
    max_items: int = 100
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    json_map: JSONFieldMap | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ConfigError("news source is missing an id")
        if not self.url:
            raise ConfigError(f"news source {self.source_id} is missing a url")
        if not 0.0 <= self.reliability <= 1.0:
            raise ConfigError(
                f"news source {self.source_id}: reliability {self.reliability} is outside [0, 1]"
            )
        if self.kind is SourceKind.JSON and self.json_map is None:
            raise ConfigError(f"news source {self.source_id}: kind=json requires a json mapping")


class NewsSource(ABC):
    """One pollable provider.

    Holds the conditional-request state for its endpoint and nothing else. A
    source is deliberately not given the HTTP client: the ingestor owns one
    shared, connection-pooled client so that fifty feeds do not open fifty
    connection pools.
    """

    __slots__ = ("_etag", "_last_modified", "spec")

    def __init__(self, spec: SourceSpec) -> None:
        self.spec = spec
        self._etag: str | None = None
        self._last_modified: str | None = None

    @property
    def source_id(self) -> str:
        return self.spec.source_id

    @property
    def name(self) -> str:
        return self.spec.name

    def request_headers(self) -> dict[str, str]:
        headers = dict(self.spec.headers)
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._last_modified:
            headers["If-Modified-Since"] = self._last_modified
        return headers

    def remember_validators(self, response: httpx.Response) -> None:
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        if etag:
            self._etag = etag
        if last_modified:
            self._last_modified = last_modified

    async def fetch(self, client: httpx.AsyncClient) -> list[RawItem]:
        """Poll once. Returns an empty list on 304, which is the common case."""
        response = await client.get(
            self.spec.url,
            headers=self.request_headers(),
            params=self.spec.params or None,
            timeout=self.spec.timeout_s,
        )
        if response.status_code == 304:
            return []
        response.raise_for_status()
        self.remember_validators(response)
        # Parsing runs in a worker thread. It is pure CPU over a payload that
        # can reach a quarter of a megabyte, and this coroutine shares its event
        # loop with the market-data websocket: measured against the live feeds,
        # doing it inline stalled the loop for 305ms in a single call, which is
        # long enough for the exchange to conclude we have stopped listening.
        # `parse` is pure and touches no shared state, so a thread is safe.
        return await asyncio.to_thread(self.parse, response)

    @abstractmethod
    def parse(self, response: httpx.Response) -> list[RawItem]:
        """Turn one successful response into raw items. Pure, and unit-testable."""


class RSSSource(NewsSource):
    """RSS/Atom via feedparser.

    feedparser is handed *bytes we already fetched* rather than a URL. Its own
    fetcher is synchronous ``urllib`` with no timeout control, which would stall
    the whole ingestion loop on one slow publisher.

    Parsing those bytes is pure CPU, and it is not cheap: a 230KB feed takes
    long enough that :meth:`NewsSource.fetch` runs it in a worker thread rather
    than on the event loop.
    """

    __slots__ = ()

    def parse(self, response: httpx.Response) -> list[RawItem]:
        import feedparser

        parsed = feedparser.parse(response.content)
        # ``bozo`` means the XML was malformed. Feeds are malformed all the
        # time and feedparser still recovers most entries, so this is a warning
        # and a quality signal, not a reason to discard the poll.
        if getattr(parsed, "bozo", 0) and not parsed.entries:
            raise NewsSourceError(
                f"{self.source_id}: feed did not parse and yielded no entries "
                f"({getattr(parsed, 'bozo_exception', None)!r})"
            )

        feed_language = (getattr(parsed, "feed", {}) or {}).get("language")
        items: list[RawItem] = []
        for entry in parsed.entries[: self.spec.max_items]:
            title = _clean_str(entry.get("title"))
            if not title:
                continue
            link = _clean_str(entry.get("link")) or None
            external_id = _clean_str(entry.get("id")) or link or title
            items.append(
                RawItem(
                    source_id=self.source_id,
                    source_name=self.name,
                    external_id=external_id,
                    title=title,
                    summary=_clean_str(entry.get("summary")),
                    body=_rss_body(entry),
                    url=link,
                    published_ms=_rss_published_ms(entry),
                    language=_clean_str(entry.get("language")) or feed_language,
                    categories=self.spec.categories + _rss_tags(entry),
                    reliability=self.spec.reliability,
                    raw={"entry_keys": sorted(entry.keys())},
                )
            )
        return items


class JSONSource(NewsSource):
    """Generic JSON API provider driven by a :class:`JSONFieldMap`.

    Covers every free government endpoint we use. An item that has no title
    after mapping is dropped rather than given a placeholder: an untitled
    article matches nothing and would only dilute the corpus statistics.
    """

    __slots__ = ()

    def parse(self, response: httpx.Response) -> list[RawItem]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise NewsSourceError(f"{self.source_id}: response was not JSON ({exc})") from exc

        fmap = self.spec.json_map
        if fmap is None:  # pragma: no cover - SourceSpec validates this
            raise NewsSourceError(f"{self.source_id}: no json mapping configured")

        raw_items = _dig(payload, fmap.items_path) if fmap.items_path else payload
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            raise NewsSourceError(
                f"{self.source_id}: items_path {fmap.items_path!r} did not resolve to a list"
            )

        items: list[RawItem] = []
        for node in raw_items[: self.spec.max_items]:
            if not isinstance(node, dict):
                continue
            title = _clean_str(_first(_dig(node, fmap.title)))
            if not title:
                continue
            url = _clean_str(_first(_dig(node, fmap.url))) or None
            external_id = _clean_str(_first(_dig(node, fmap.external_id))) or url or title
            published_raw = _first(_dig(node, fmap.published)) if fmap.published else None
            items.append(
                RawItem(
                    source_id=self.source_id,
                    source_name=self.name,
                    external_id=external_id,
                    title=title,
                    summary=_clean_str(_first(_dig(node, fmap.summary))),
                    body=_clean_str(_first(_dig(node, fmap.body))),
                    url=url,
                    published_ms=_parse_time_ms(published_raw, fmap.published_format),
                    language=_clean_str(_first(_dig(node, fmap.language))) or None,
                    categories=self.spec.categories + _as_tuple(_dig(node, fmap.categories)),
                    reliability=self.spec.reliability,
                    raw=node,
                )
            )
        return items


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _dig(node: Any, path: str) -> Any:
    """Resolve a dotted path with integer indices and ``*`` list mapping."""
    if not path:
        return None
    current: Any = node
    segments = path.split(".")
    for index, segment in enumerate(segments):
        if current is None:
            return None
        if segment == "*":
            if not isinstance(current, list):
                return None
            rest = ".".join(segments[index + 1 :])
            if not rest:
                return current
            out = [_dig(child, rest) for child in current]
            return [value for value in out if value is not None]
        if segment.isdigit() and isinstance(current, list):
            position = int(segment)
            current = current[position] if position < len(current) else None
            continue
        if isinstance(current, dict):
            current = current.get(segment)
            continue
        return None
    return current


def _first(value: Any) -> Any:
    """Collapse a ``*``-mapped list to its first element."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(v) for v in value if v is not None)
    return (str(value),)


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------


def _parse_time_ms(value: Any, fmt: str) -> int | None:
    """Parse a source timestamp to epoch milliseconds, or ``None``.

    Returning ``None`` is always acceptable -- ``published_ms`` is metadata, and
    the pipeline's ordering guarantee comes from ``ingested_ms``. Guessing a
    time we do not have would be worse than admitting we do not have it.
    """
    if value is None or value == "":
        return None
    try:
        if fmt == "unix":
            return int(float(value) * 1000)
        if fmt == "unix_ms":
            return int(value)
        if fmt == "date":
            # A bare date. Midnight UTC is the only defensible reading, and the
            # resulting several-hour error is exactly why backtests filter on
            # ingestion time instead.
            return int(parse_rfc3339(f"{value}T00:00:00Z").timestamp() * 1000)
        if fmt in ("rfc3339", "iso8601"):
            return int(parse_rfc3339(str(value)).timestamp() * 1000)
    except (TypeError, ValueError) as exc:
        log.debug("news.source.time_parse_failed", value=repr(value), fmt=fmt, error=str(exc))
        return None

    # auto
    text = str(value).strip()
    if text.isdigit():
        number = int(text)
        return number * 1000 if number < 100_000_000_000 else number
    try:
        return int(parse_rfc3339(text).timestamp() * 1000)
    except ValueError:
        if len(text) == 10 and text[4] == "-":
            return _parse_time_ms(text, "date")
        log.debug("news.source.time_unparsed", value=text)
        return None


def _rss_published_ms(entry: Any) -> int | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        struct = entry.get(key)
        if struct:
            # feedparser normalises to UTC, so timegm (not mktime) is correct.
            return calendar.timegm(struct) * 1000
    for key in ("published", "updated"):
        text = _clean_str(entry.get(key))
        if text:
            return _parse_time_ms(text, "auto")
    return None


def _rss_body(entry: Any) -> str:
    content = entry.get("content")
    if isinstance(content, list) and content:
        value = content[0].get("value") if isinstance(content[0], dict) else None
        if value:
            return str(value)
    return _clean_str(entry.get("summary"))


def _rss_tags(entry: Any) -> tuple[str, ...]:
    tags = entry.get("tags")
    if not isinstance(tags, list):
        return ()
    out = [_clean_str(tag.get("term")) for tag in tags if isinstance(tag, dict)]
    return tuple(t for t in out if t)


# ---------------------------------------------------------------------------
# Construction from configuration
# ---------------------------------------------------------------------------


def build_source(spec: SourceSpec) -> NewsSource:
    if spec.kind is SourceKind.RSS:
        return RSSSource(spec)
    if spec.kind is SourceKind.JSON:
        return JSONSource(spec)
    raise ConfigError(f"unsupported news source kind: {spec.kind}")  # pragma: no cover


def load_source_specs(path: str | Path | None = None) -> list[SourceSpec]:
    """Read ``configs/news_sources.yaml``.

    A missing file is a configuration error rather than an empty list: news
    ingestion that silently does nothing is the failure mode most likely to go
    unnoticed for a week. Callers that want news to be optional check
    ``settings.news.enabled`` instead.
    """
    import yaml

    resolved = Path(path or DEFAULT_SOURCES_FILE)
    if not resolved.is_absolute():
        resolved = _REPO_ROOT / resolved
    if not resolved.exists():
        raise ConfigError(f"news sources file not found: {resolved}")

    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{resolved} must contain a YAML mapping at the top level")

    defaults = loaded.get("defaults") or {}
    entries = loaded.get("sources") or []
    if not isinstance(entries, list):
        raise ConfigError(f"{resolved}: 'sources' must be a list")

    specs: list[SourceSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ConfigError(f"{resolved}: every source entry must be a mapping")
        spec = _spec_from_mapping({**defaults, **entry})
        if spec.source_id in seen:
            raise ConfigError(f"{resolved}: duplicate news source id {spec.source_id!r}")
        seen.add(spec.source_id)
        specs.append(spec)
    return specs


def _spec_from_mapping(data: dict[str, Any]) -> SourceSpec:
    json_block = data.get("json")
    json_map: JSONFieldMap | None = None
    if json_block is not None:
        if not isinstance(json_block, dict):
            raise ConfigError(f"news source {data.get('id')!r}: 'json' must be a mapping")
        fields = json_block.get("fields") or {}
        json_map = JSONFieldMap(
            items_path=str(json_block.get("items_path", "")),
            external_id=str(fields.get("external_id", "id")),
            title=str(fields.get("title", "title")),
            summary=str(fields.get("summary", "")),
            body=str(fields.get("body", "")),
            url=str(fields.get("url", "")),
            published=str(fields.get("published", "")),
            published_format=str(json_block.get("published_format", "auto")),
            categories=str(fields.get("categories", "")),
            language=str(fields.get("language", "")),
        )

    kind_raw = str(data.get("kind", "rss")).lower()
    try:
        kind = SourceKind(kind_raw)
    except ValueError as exc:
        raise ConfigError(f"unknown news source kind {kind_raw!r}") from exc

    return SourceSpec(
        source_id=str(data.get("id", "")),
        name=str(data.get("name", data.get("id", ""))),
        url=str(data.get("url", "")),
        kind=kind,
        enabled=bool(data.get("enabled", True)),
        reliability=float(data.get("reliability", 0.5)),
        categories=tuple(str(c) for c in (data.get("categories") or ())),
        min_interval_s=float(data.get("min_interval_s", 60.0)),
        timeout_s=float(data.get("timeout_s", 15.0)),
        max_items=int(data.get("max_items", 100)),
        headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
        params={str(k): str(v) for k, v in (data.get("params") or {}).items()},
        json_map=json_map,
    )


def load_sources(path: str | Path | None = None, *, include_disabled: bool = False) -> list[NewsSource]:
    specs = load_source_specs(path)
    return [build_source(s) for s in specs if s.enabled or include_disabled]


from kat.news.types import RawItem  # noqa: E402  - placed last to keep the reading order above
