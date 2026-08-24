"""Turn a provider's item into an :class:`~kat.news.types.Article`.

Two jobs, and the second is the one that matters.

**Cleaning.** Strip the HTML, entity-decode, collapse whitespace, drop the
boilerplate that wire services append to everything. Dull but necessary: an
unstripped ``<p>`` tag changes a content hash, so two copies of one story would
otherwise fail to deduplicate on a formatting difference.

**Stamping the two timestamps.** ``published_ms`` is what the source claims;
``ingested_ms`` is when *this process* could first have acted on it. Only the
second is ever admissible in a backtest, and this module is where the second one
is set -- from the platform clock, once, at the moment the item arrives.

That distinction is not theoretical. Several government feeds publish a
document's *issue date* rather than the moment it became fetchable, so their
items routinely arrive hours "old". Filtering a simulation on ``published_ms``
would hand a strategy those hours as hindsight.

Fingerprints
------------
Each article gets two:

``content_hash``
    SHA-256 of the normalised title and body. Exact-match deduplication. Derived
    from content rather than assigned randomly, so re-ingesting the same story
    after a restart produces the same article id and collides with the existing
    row instead of quietly creating a second copy.

``simhash``
    A 64-bit locality-sensitive fingerprint. Near-duplicate detection, for the
    case that actually dominates a news feed: the same wire story reprinted by
    six outlets with a different headline and one added sentence. An exact hash
    sees six unrelated stories; simhash sees one story six times, which is the
    difference between "six sources corroborate this" and "six things happened".
"""

from __future__ import annotations

import hashlib
import html
import itertools
import re
import unicodedata

from kat.common.enums import DataQuality
from kat.common.timeutil import Clock
from kat.news.types import Article, RawItem, article_id_for_hash
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["content_hash", "normalise", "simhash", "simhash_distance"]

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9']+")

#: Trailing furniture wire services attach to every item. Removed before
#: hashing, or two copies of one story differ only by their footer.
_BOILERPLATE = (
    re.compile(r"\(reporting by [^)]*\)", re.I),
    re.compile(r"\(editing by [^)]*\)", re.I),
    re.compile(r"all rights reserved\.?", re.I),
    re.compile(r"copyright \d{4}[^.]*\.", re.I),
    re.compile(r"this material may not be published[^.]*\.", re.I),
    re.compile(r"click here to (?:read|subscribe)[^.]*\.", re.I),
    re.compile(r"follow us on [a-z]+\.?", re.I),
)

#: Very common words carry no signal for a fingerprint and, being present in
#: everything, actively pull unrelated documents together.
_STOPWORDS = frozenset(
    """a an the and or but if while of to in on at by for with from as is are was
    were be been being it its this that these those he she they we you i not no
    will would can could may might must shall should do does did have has had
    said says say new more than after before over under about into out up down
    who whom which what when where why how""".split()  # noqa: SIM905 - a prose word list stays readable; a 77-element list literal does not
)


def clean_text(value: str) -> str:
    """HTML out, entities decoded, whitespace collapsed, boilerplate removed."""
    if not value:
        return ""
    text = html.unescape(value)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)  # feeds double-escape surprisingly often
    text = unicodedata.normalize("NFKC", text)
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenise(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def content_hash(title: str, body: str) -> str:
    """Stable SHA-256 over the normalised text.

    Hashes the *token sequence* rather than the raw string, so a difference in
    punctuation or capitalisation does not defeat exact-match deduplication.
    """
    tokens = tokenise(f"{title} {body}")
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def simhash(text: str, *, bits: int = 64) -> int:
    """Charikar simhash over token shingles.

    Each token votes on every bit of the fingerprint, weighted by frequency; the
    output bit is 1 where the votes are net positive. Documents differing in a
    few tokens keep almost all their bits, so Hamming distance measures textual
    similarity directly.

    Shingles of two adjacent tokens rather than single tokens, because word order
    carries most of what distinguishes two stories about the same subject.
    """
    tokens = tokenise(text)
    if not tokens:
        return 0
    shingles = (
        [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        if len(tokens) > 1
        else tokens
    )

    vector = [0] * bits
    for shingle in shingles:
        digest = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(bits):
            vector[bit] += 1 if (digest >> bit) & 1 else -1

    fingerprint = 0
    for bit in range(bits):
        if vector[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def simhash_distance(left: int, right: int) -> int:
    """Hamming distance between two fingerprints, in bits."""
    return ((left ^ right).bit_count()) if left and right else 64


def normalise(item: RawItem, clock: Clock) -> Article | None:
    """Build an :class:`Article`, or ``None`` if the item carries no usable text.

    The ingestion timestamp is taken from the platform clock **here**, once. It
    is deliberately not taken later in the pipeline: an article that spends two
    seconds in extraction did not become knowable two seconds later.
    """
    title = clean_text(item.title)
    if not title:
        log.debug("news.item_discarded", source=item.source_id, reason="no title")
        return None

    summary = clean_text(item.summary)
    body = clean_text(item.body) or summary

    # A title alone can still be a usable signal -- a headline is often the whole
    # story for a market -- but a title of three words is not.
    if len(tokenise(f"{title} {body}")) < 3:
        log.debug("news.item_discarded", source=item.source_id, reason="too little text")
        return None

    digest = content_hash(title, body)
    ingested_ms = clock.now_ms()

    quality = DataQuality.OK
    if item.published_ms is None:
        # No claimed publication time. Usable, but flagged: the freshness signal
        # a strategy reads is weaker for it.
        quality = DataQuality.PARTIAL
    elif item.published_ms > ingested_ms + 3_600_000:
        # Claimed to be published more than an hour in the future. A clock or
        # timezone bug at the source; the item is kept and marked rather than
        # trusted.
        quality = DataQuality.SUSPECT
        log.warning(
            "news.future_publication_time",
            source=item.source_id,
            published_ms=item.published_ms,
            ingested_ms=ingested_ms,
            ahead_ms=item.published_ms - ingested_ms,
        )

    return Article(
        article_id=article_id_for_hash(digest),
        source=item.source_id,
        source_name=item.source_name,
        title=title,
        summary=summary[:2000],
        body=body[:20_000],
        content_hash=digest,
        simhash=simhash(f"{title} {body}"),
        ingested_ms=ingested_ms,
        published_ms=item.published_ms,
        url=item.url,
        external_id=item.external_id or None,
        language=item.language or "en",
        source_reliability=item.reliability,
        categories=item.categories,
        quality=quality,
    )
