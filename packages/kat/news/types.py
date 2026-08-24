"""The vocabulary the news pipeline stages hand to one another.

The pipeline is a straight line -- source, ingest, dedupe, extract, match,
impact -- and every stage needs to name the object the previous stage produced.
If each stage owned its own output type the imports would form a cycle the
moment :mod:`kat.news.matching` wanted to read what :mod:`kat.news.extract`
attached to an article. The value types therefore live here and the stages
import downwards only.

Two timestamps, and why both are load-bearing
---------------------------------------------
:class:`Article` carries ``published_ms`` (what the source claims) and
``ingested_ms`` (when this process could first have acted on it). **Only the
second is admissible in a backtest.** This is not pedantry: several of the free
government feeds back-date items by hours, because the timestamp they publish is
the issue date of the document rather than the moment it became fetchable. A
simulator that filtered on ``published_ms`` would hand a strategy several hours
of hindsight and produce a beautiful, entirely fictitious equity curve.
``ingested_ms`` comes from the platform :class:`~kat.common.timeutil.Clock`, so
a replay reproduces it exactly.

What is deliberately not persisted
----------------------------------
``keywords`` are TF-IDF weights against a *rolling* corpus. Writing them to the
database would freeze a vocabulary fitted on the whole corpus -- including
documents from the future -- into a row that a backtest then reads as if it had
been knowable at ingest time. They are recomputed at query time from a
point-in-time corpus instead, and never stored. The ``claims`` column is
likewise left NULL: we do not do claim extraction (see :mod:`kat.news.extract`
for an honest account of what rule-based NLP can and cannot do), and filling the
column with something that is not a claim would be worse than leaving it empty.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kat.common.enums import DataQuality
from kat.common.timeutil import from_ms
from kat.db.models import NewsArticle

__all__ = [
    "Article",
    "Entity",
    "EntityKind",
    "EventType",
    "ExtractedEvent",
    "RawItem",
    "article_id_for_hash",
]


class EntityKind(str, Enum):
    """Coarse entity categories the rule-based extractor can actually support.

    Deliberately coarse. A statistical extractor with no trained model cannot
    reliably separate, say, a person from a fictional character, so it does not
    pretend to: these are the categories a gazetteer plus a handful of regular
    expressions get right most of the time.
    """

    PERSON = "person"
    ORG = "org"
    GOV_BODY = "gov_body"
    COUNTRY = "country"
    REGION = "region"
    CITY = "city"
    TICKER = "ticker"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    MONEY = "money"
    PERCENT = "percent"
    NUMBER = "number"
    DATE = "date"
    STORM = "storm"
    OTHER = "other"


class EventType(str, Enum):
    """Event classes with a cue-phrase lexicon behind them.

    Each value corresponds to a group of trigger phrases in
    :mod:`kat.news.extract` and to a base magnitude in :mod:`kat.news.impact`.
    ``UNKNOWN`` is the honest answer when no cue fires, and it is common.
    """

    ELECTION_RESULT = "election_result"
    POLL_RELEASE = "poll_release"
    APPOINTMENT = "appointment"
    RESIGNATION = "resignation"
    LEGAL_ACTION = "legal_action"
    LEGISLATION = "legislation"
    REGULATION = "regulation"
    MONETARY_POLICY = "monetary_policy"
    ECONOMIC_RELEASE = "economic_release"
    EARNINGS = "earnings"
    MERGER = "merger"
    CONFLICT = "conflict"
    DIPLOMACY = "diplomacy"
    DISASTER = "disaster"
    WEATHER_ALERT = "weather_alert"
    HEALTH = "health"
    SPORTS_RESULT = "sports_result"
    LAUNCH = "launch"
    DEATH = "death"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Entity:
    """One entity mention, already folded across repeated mentions in a document.

    ``salience`` is a within-document weight in [0, 1]: it combines mention count
    with position, because an entity named in the headline is what the story is
    about and an entity named once in the last paragraph usually is not.
    """

    text: str
    kind: EntityKind
    normalised: str
    count: int = 1
    salience: float = 0.0
    in_title: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind.value,
            "normalised": self.normalised,
            "count": self.count,
            "salience": round(self.salience, 4),
            "in_title": self.in_title,
        }


@dataclass(frozen=True, slots=True)
class ExtractedEvent:
    """A cue-phrase match promoted to a typed event.

    ``polarity`` is relative to ``subject`` and takes values -1, 0 or +1: "wins"
    is +1 for its subject, "resigns" is -1, "meets with" is 0. It is *not* a
    market direction -- turning it into one requires knowing which side of a
    market the subject sits on, which is :mod:`kat.news.impact`'s job and is
    frequently impossible.
    """

    event_type: EventType
    trigger: str
    subject: str | None
    polarity: int
    confidence: float
    span: tuple[int, int]
    in_title: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "trigger": self.trigger,
            "subject": self.subject,
            "polarity": self.polarity,
            "confidence": round(self.confidence, 4),
            "span": list(self.span),
            "in_title": self.in_title,
        }


@dataclass(frozen=True, slots=True)
class RawItem:
    """One item exactly as a provider produced it, before normalisation.

    ``raw`` retains the provider payload so that a parsing bug is diagnosed from
    what we actually received rather than from what we managed to keep.
    """

    source_id: str
    source_name: str
    external_id: str
    title: str
    summary: str = ""
    body: str = ""
    url: str | None = None
    published_ms: int | None = None
    language: str | None = None
    categories: tuple[str, ...] = ()
    reliability: float = 0.5
    raw: dict[str, Any] = field(default_factory=dict)


def article_id_for_hash(content_hash: str) -> str:
    """Derive the article id from the content hash.

    Content-derived rather than random so that re-ingesting the same story after
    a restart produces the same id and collides with the existing row's unique
    constraint, instead of silently creating a second copy of one event.
    """
    return f"art_{content_hash[:40]}"


@dataclass(frozen=True, slots=True)
class Article:
    """A normalised news item at whatever stage of enrichment it has reached.

    Frozen: each pipeline stage returns a new copy via :meth:`evolve` rather
    than mutating in place, so an article handed to two consumers cannot acquire
    a third consumer's annotations behind their backs.
    """

    article_id: str
    source: str
    source_name: str
    title: str
    summary: str
    body: str
    content_hash: str
    simhash: int
    ingested_ms: int
    published_ms: int | None = None
    url: str | None = None
    external_id: str | None = None
    language: str = "en"
    source_reliability: float = 0.5
    categories: tuple[str, ...] = ()
    entities: tuple[Entity, ...] = ()
    events: tuple[ExtractedEvent, ...] = ()
    keywords: tuple[tuple[str, float], ...] = ()
    relevance: float | None = None
    quality: DataQuality = DataQuality.OK
    duplicate_of: str | None = None
    corroborations: int = 0

    # -- point-in-time -------------------------------------------------------

    def knowable_at(self, now_ms: int) -> bool:
        """Whether a decision made at ``now_ms`` was allowed to see this article."""
        return self.ingested_ms <= now_ms

    def age_ms(self, now_ms: int) -> int:
        """Age measured from ingestion, never from the source's claimed time."""
        return max(0, now_ms - self.ingested_ms)

    @property
    def publication_lag_ms(self) -> int | None:
        """How far behind the source's claimed time we were.

        Large positive values are the signature of a back-dating feed, and are
        why ``published_ms`` is never used for filtering.
        """
        if self.published_ms is None:
            return None
        return self.ingested_ms - self.published_ms

    # -- enrichment ----------------------------------------------------------

    def evolve(self, **changes: Any) -> Article:
        return dataclasses.replace(self, **changes)

    @property
    def text(self) -> str:
        """Title plus body, the form every text stage consumes."""
        return f"{self.title}\n{self.body}" if self.body else self.title

    def entity_keys(self) -> frozenset[str]:
        return frozenset(e.normalised for e in self.entities)

    def has_event(self, event_type: EventType) -> bool:
        return any(e.event_type is event_type for e in self.events)

    # -- persistence ---------------------------------------------------------

    def to_orm(self) -> NewsArticle:
        """Build the row. See the module docstring for what is left out and why."""
        published = None if self.published_ms is None else from_ms(self.published_ms)
        return NewsArticle(
            article_id=self.article_id,
            content_hash=self.content_hash,
            url=self.url,
            title=self.title,
            summary=self.summary,
            published_ts=published,
            language=self.language,
            source_reliability=self.source_reliability,
            entities=[e.to_dict() for e in self.entities],
            extracted_events=[e.to_dict() for e in self.events],
            claims=None,
            relevance=self.relevance,
            source=self.source,
            source_id=self.external_id,
            source_ts=published,
            ingested_ts=from_ms(self.ingested_ms),
            quality=self.quality.value,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for logs, the dashboard and LLM context blocks."""
        return {
            "article_id": self.article_id,
            "source": self.source,
            "source_name": self.source_name,
            "title": self.title,
            "url": self.url,
            "published_ms": self.published_ms,
            "ingested_ms": self.ingested_ms,
            "language": self.language,
            "source_reliability": round(self.source_reliability, 4),
            "categories": list(self.categories),
            "entities": [e.to_dict() for e in self.entities],
            "events": [e.to_dict() for e in self.events],
            "keywords": [[t, round(w, 4)] for t, w in self.keywords],
            "relevance": self.relevance,
            "quality": self.quality.value,
            "duplicate_of": self.duplicate_of,
            "corroborations": self.corroborations,
        }
