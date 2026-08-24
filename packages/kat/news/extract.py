"""Entity and event extraction without a paid NLP service.

What this can and cannot do, stated up front so nobody mistakes it for
understanding:

**It can** find capitalised name spans, recognise known organisations and
government bodies from a gazetteer, pull out numbers, money, percentages and
dates, spot tickers and commodities, and match cue phrases ("wins", "resigns",
"raises rates") to a typed event with a polarity. It weights entities by where
and how often they appear, so the subject of a headline outranks a name
mentioned once at the end.

**It cannot** resolve coreference, parse syntax, handle negation reliably, or
tell you that "the central bank" and "the Fed" are the same institution unless
the gazetteer says so. It will occasionally decide a sentence-initial word is a
person.

That is an acceptable trade. The consumer is
:mod:`kat.news.matching`, which scores *relevance* and hands the result to a
strategy that already refuses to act on anything below a confidence threshold.
Precision matters more than recall here, and the design leans that way: a cue
phrase must appear near its subject, and an unmatched article is a normal
outcome.

Keywords use TF-IDF over a rolling background corpus, which is genuinely
statistical rather than a list -- it learns what is unusual about *this* article
relative to recent news, which is what makes a term worth matching on.
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from typing import Any

from kat.news.normalise import tokenise
from kat.news.types import Article, Entity, EntityKind, EventType, ExtractedEvent
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["Extractor", "extract_entities", "extract_events"]


# --- gazetteers -------------------------------------------------------------
# Small and deliberately high-precision. A name here is one whose appearance in
# a story genuinely tells you what the story is about.

_GOV_BODIES = {
    "federal reserve": "federal_reserve",
    "the fed": "federal_reserve",
    "fed": "federal_reserve",
    "fomc": "federal_reserve",
    "european central bank": "ecb",
    "ecb": "ecb",
    "bank of england": "boe",
    "bank of japan": "boj",
    "treasury": "us_treasury",
    "congress": "us_congress",
    "senate": "us_senate",
    "house of representatives": "us_house",
    "supreme court": "us_supreme_court",
    "white house": "white_house",
    "pentagon": "us_defense",
    "state department": "us_state_dept",
    "justice department": "us_doj",
    "doj": "us_doj",
    "sec": "us_sec",
    "cftc": "us_cftc",
    "fda": "us_fda",
    "cdc": "us_cdc",
    "nasa": "nasa",
    "noaa": "noaa",
    "national weather service": "nws",
    "national hurricane center": "nhc",
    "bureau of labor statistics": "bls",
    "bls": "bls",
    "united nations": "un",
    "nato": "nato",
    "opec": "opec",
    "world health organization": "who",
}

_COUNTRIES = {
    "united states", "u.s.", "us", "usa", "america", "china", "russia", "ukraine",
    "israel", "iran", "india", "japan", "germany", "france", "united kingdom",
    "uk", "britain", "canada", "mexico", "brazil", "taiwan", "north korea",
    "south korea", "saudi arabia", "turkey", "egypt", "venezuela", "argentina",
    "australia", "italy", "spain", "poland", "netherlands", "switzerland",
}

_COMMODITIES = {
    "gold", "silver", "oil", "crude", "brent", "wti", "natural gas", "copper",
    "wheat", "corn", "soybeans", "bitcoin", "ethereum", "uranium", "platinum",
}

_CURRENCIES = {"dollar", "euro", "yen", "pound", "yuan", "renminbi", "franc", "rupee"}

# --- event cues -------------------------------------------------------------
# (pattern, event type, polarity relative to the subject). Polarity is about the
# subject, not about any market: "wins" is +1 for whoever won, and turning that
# into a direction requires knowing which side of a market they sit on.

_EVENT_CUES: tuple[tuple[re.Pattern[str], EventType, int], ...] = (
    (re.compile(r"\bwins?\b|\bwon\b|\bvictor(?:y|ious)\b|\bdefeats?\b", re.I), EventType.ELECTION_RESULT, 1),
    (re.compile(r"\bloses?\b|\blost\b|\bconcedes?\b|\bdefeated\b", re.I), EventType.ELECTION_RESULT, -1),
    (re.compile(r"\bpoll\b|\bpolling\b|\bsurvey (?:shows?|finds?)\b", re.I), EventType.POLL_RELEASE, 0),
    (re.compile(r"\bappoints?\b|\bnominates?\b|\bnamed? (?:as )?(?:the )?new\b", re.I), EventType.APPOINTMENT, 1),
    (re.compile(r"\bresigns?\b|\bsteps? down\b|\bquits?\b|\bousted\b|\bfired\b", re.I), EventType.RESIGNATION, -1),
    (re.compile(r"\bsues?\b|\blawsuit\b|\bindicted?\b|\bcharged with\b|\bconvicted\b", re.I), EventType.LEGAL_ACTION, -1),
    (re.compile(r"\bpasses? (?:the )?(?:bill|law|act)\b|\bsigns? into law\b|\benacts?\b", re.I), EventType.LEGISLATION, 1),
    (re.compile(r"\bvetoes?\b|\bblocks? (?:the )?(?:bill|law)\b|\brejects? (?:the )?bill\b", re.I), EventType.LEGISLATION, -1),
    (re.compile(r"\bregulat(?:es?|ion|ory)\b|\bban(?:s|ned)?\b|\brule(?:s|d)? that\b", re.I), EventType.REGULATION, 0),
    (re.compile(r"\braises? rates?\b|\bhikes? rates?\b|\brate (?:hike|increase)\b", re.I), EventType.MONETARY_POLICY, 1),
    (re.compile(r"\bcuts? rates?\b|\blowers? rates?\b|\brate cut\b", re.I), EventType.MONETARY_POLICY, -1),
    (re.compile(r"\bholds? rates?\b|\bkeeps? rates? (?:un)?changed\b|\bleaves? rates?\b", re.I), EventType.MONETARY_POLICY, 0),
    (re.compile(r"\b(?:cpi|inflation|gdp|unemployment|payrolls|jobs report)\b", re.I), EventType.ECONOMIC_RELEASE, 0),
    (re.compile(r"\bearnings\b|\bquarterly results\b|\bbeats? estimates?\b|\bmisses? estimates?\b", re.I), EventType.EARNINGS, 0),
    (re.compile(r"\bacquires?\b|\bmerger\b|\bbuyout\b|\btakeover\b|\bacquisition\b", re.I), EventType.MERGER, 0),
    (re.compile(r"\bstrikes?\b|\battacks?\b|\binvades?\b|\bair ?strikes?\b|\bmissile\b|\bshelling\b", re.I), EventType.CONFLICT, -1),
    (re.compile(r"\bceasefire\b|\btruce\b|\bpeace (?:deal|talks|agreement)\b|\bsummit\b", re.I), EventType.DIPLOMACY, 1),
    (re.compile(r"\bearthquake\b|\bhurricane\b|\bwildfire\b|\bflood(?:ing|s)?\b|\btornado\b", re.I), EventType.DISASTER, -1),
    (re.compile(r"\bwarning (?:issued|in effect)\b|\bwatch (?:issued|in effect)\b|\badvisory\b", re.I), EventType.WEATHER_ALERT, 0),
    (re.compile(r"\boutbreak\b|\bepidemic\b|\bpandemic\b|\brecall(?:s|ed)?\b", re.I), EventType.HEALTH, -1),
    (re.compile(r"\bdefeats?\b|\bbeat\b|\bfinal score\b|\bchampionship\b|\btitle\b", re.I), EventType.SPORTS_RESULT, 0),
    (re.compile(r"\blaunch(?:es|ed)?\b|\bliftoff\b|\bdocked?\b", re.I), EventType.LAUNCH, 1),
    (re.compile(r"\bdies?\b|\bdied\b|\bdeath of\b|\bpasses? away\b", re.I), EventType.DEATH, -1),
)

# Both apostrophe forms are matched on purpose: real headlines use the
# typographic right single quote, and a pattern accepting only the ASCII
# apostrophe splits such a name into two fragments and loses it.
_PROPER_NOUN = re.compile(
    r"\b(?:[A-Z][a-z'\u2019]+(?:\s+(?:of|the|de|van|von))?\s*){1,4}[A-Z][a-z'\u2019]+\b|\b[A-Z][a-z'\u2019]{2,}\b"
)
_TICKER = re.compile(r"\b[A-Z]{2,5}\b(?=\s*(?:shares?|stock|Inc|Corp|\)))")
_MONEY = re.compile(r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|trillion|bn|mn|k)?", re.I)
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent|percentage points?|bps|basis points?)\b", re.I)
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_DATE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}"
    r"(?:,?\s+\d{4})?\b|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)

#: Words that start sentences and get mistaken for names.
_SENTENCE_STARTERS = frozenset(
    """the a an this that these those he she it they we i but and or if when while
    after before however meanwhile according following despite although since
    new more most many some several other another such then there here what""".split()  # noqa: SIM905 - a prose word list stays readable; a 40-element list literal does not
)


def _normalise_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def extract_entities(title: str, body: str, *, max_entities: int = 25) -> tuple[Entity, ...]:
    """Find entities and weight them by prominence.

    Salience combines mention count with position: an entity in the headline is
    what the story is about; one mentioned once near the end usually is not. That
    weighting is what stops a passing reference to a country from making an
    article look relevant to every market about that country.
    """
    text = f"{title}. {body}"
    lowered = text.lower()
    found: dict[str, Entity] = {}

    def add(surface: str, kind: EntityKind, key: str | None = None) -> None:
        surface = surface.strip()
        if not surface:
            return
        normalised = key or _normalise_key(surface)
        if not normalised:
            return
        in_title = surface.lower() in title.lower()
        existing = found.get(normalised)
        if existing is None:
            found[normalised] = Entity(
                text=surface, kind=kind, normalised=normalised, count=1, in_title=in_title
            )
        else:
            found[normalised] = Entity(
                text=existing.text,
                kind=existing.kind,
                normalised=normalised,
                count=existing.count + 1,
                in_title=existing.in_title or in_title,
            )

    # Gazetteer hits first: these are the high-precision ones.
    for phrase, key in _GOV_BODIES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            add(phrase, EntityKind.GOV_BODY, key)
    for country in _COUNTRIES:
        if re.search(rf"\b{re.escape(country)}\b", lowered):
            add(country, EntityKind.COUNTRY, _normalise_key(country))
    for commodity in _COMMODITIES:
        if re.search(rf"\b{re.escape(commodity)}\b", lowered):
            add(commodity, EntityKind.COMMODITY, _normalise_key(commodity))
    for currency in _CURRENCIES:
        if re.search(rf"\b{re.escape(currency)}\b", lowered):
            add(currency, EntityKind.CURRENCY, _normalise_key(currency))

    # Proper-noun spans, minus the sentence-initial false positives.
    for match in _PROPER_NOUN.finditer(text):
        surface = match.group(0).strip()
        if len(surface) < 3:
            continue
        first = surface.split()[0].lower()
        at_sentence_start = match.start() == 0 or text[max(0, match.start() - 2)] in ".!?"
        if at_sentence_start and first in _SENTENCE_STARTERS:
            continue
        key = _normalise_key(surface)
        if key in found or key in _GOV_BODIES.values():
            continue
        # Two or more capitalised words reads as an organisation more often than
        # a person; a single one is more often a person or place.
        kind = EntityKind.ORG if len(surface.split()) > 1 else EntityKind.PERSON
        add(surface, kind)

    for pattern, kind in (
        (_TICKER, EntityKind.TICKER),
        (_MONEY, EntityKind.MONEY),
        (_PERCENT, EntityKind.PERCENT),
        (_DATE, EntityKind.DATE),
    ):
        for match in pattern.finditer(text):
            add(match.group(0), kind)

    if not found:
        return ()

    # Salience: mentions, log-damped, with a strong bonus for the headline.
    peak = max(e.count for e in found.values())
    scored: list[Entity] = []
    for entity in found.values():
        frequency = math.log1p(entity.count) / math.log1p(peak) if peak > 1 else 1.0
        salience = min(1.0, 0.6 * frequency + (0.4 if entity.in_title else 0.0))
        scored.append(
            Entity(
                text=entity.text,
                kind=entity.kind,
                normalised=entity.normalised,
                count=entity.count,
                salience=salience,
                in_title=entity.in_title,
            )
        )
    scored.sort(key=lambda e: (-e.salience, -e.count, e.normalised))
    return tuple(scored[:max_entities])


def extract_events(
    title: str, body: str, entities: tuple[Entity, ...]
) -> tuple[ExtractedEvent, ...]:
    """Match cue phrases to typed events, attributing a subject where possible.

    A cue is only attributed to a subject that appears **near** it -- within a
    window of characters -- because "Smith criticised the decision to fire Jones"
    should not record Smith as resigning. Where no entity is near enough, the
    event is recorded with a null subject rather than the most prominent entity,
    since a confident wrong attribution is worse than an honest absent one.
    """
    text = f"{title}. {body}"
    title_length = len(title) + 1
    events: list[ExtractedEvent] = []
    seen: set[tuple[EventType, str]] = set()

    #: How close an entity must be to a cue to be treated as its subject.
    proximity = 120

    entity_positions: list[tuple[str, int]] = []
    for entity in entities:
        if entity.kind in (EntityKind.MONEY, EntityKind.PERCENT, EntityKind.NUMBER, EntityKind.DATE):
            continue
        for match in re.finditer(re.escape(entity.text), text, re.I):
            entity_positions.append((entity.normalised, match.start()))

    for pattern, event_type, polarity in _EVENT_CUES:
        for match in pattern.finditer(text):
            key = (event_type, match.group(0).lower())
            if key in seen:
                continue
            seen.add(key)

            start = match.start()
            in_title = start < title_length

            subject = None
            best_distance = proximity + 1
            for normalised, position in entity_positions:
                distance = abs(position - start)
                if distance < best_distance:
                    subject, best_distance = normalised, distance
            if best_distance > proximity:
                subject = None

            # A cue in the headline is what the story is about; one buried in
            # paragraph nine usually is not, and an unattributed cue is weaker
            # still.
            confidence = 0.5
            if in_title:
                confidence += 0.3
            if subject is not None:
                confidence += 0.2 * (1 - best_distance / proximity)

            events.append(
                ExtractedEvent(
                    event_type=event_type,
                    trigger=match.group(0),
                    subject=subject,
                    polarity=polarity,
                    confidence=min(1.0, confidence),
                    span=(start, match.end()),
                    in_title=in_title,
                )
            )

    events.sort(key=lambda e: (-e.confidence, e.span[0]))
    return tuple(events[:12])


class Extractor:
    """Entity, event and keyword extraction with a rolling IDF background.

    The background corpus is what makes the keywords meaningful: a term is
    interesting because it is rare *in recent news*, not because it is rare in
    English. "Hurricane" is unremarkable during hurricane season and highly
    informative in March, and only a rolling corpus can tell the difference.
    """

    def __init__(self, *, background_size: int = 2_000, max_keywords: int = 12) -> None:
        self.max_keywords = max_keywords
        self._background: deque[frozenset[str]] = deque(maxlen=background_size)
        self._document_frequency: Counter[str] = Counter()

    def _observe(self, tokens: frozenset[str]) -> None:
        if len(self._background) == self._background.maxlen:
            evicted = self._background[0]
            self._document_frequency.subtract(evicted)
            # Counter.subtract leaves zeros behind, which would grow unbounded.
            for token in evicted:
                if self._document_frequency[token] <= 0:
                    del self._document_frequency[token]
        self._background.append(tokens)
        self._document_frequency.update(tokens)

    def keywords(self, text: str) -> tuple[tuple[str, float], ...]:
        """TF-IDF keywords against the rolling background."""
        tokens = tokenise(text)
        if not tokens:
            return ()
        counts = Counter(tokens)
        unique = frozenset(counts)
        corpus_size = max(1, len(self._background))

        scored: list[tuple[str, float]] = []
        for token, count in counts.items():
            frequency = count / len(tokens)
            document_frequency = self._document_frequency.get(token, 0)
            idf = math.log((corpus_size + 1) / (document_frequency + 1)) + 1.0
            scored.append((token, frequency * idf))

        self._observe(unique)
        scored.sort(key=lambda kv: -kv[1])
        return tuple(scored[: self.max_keywords])

    def enrich(self, article: Article) -> Article:
        """Attach entities, events and keywords to an article."""
        entities = extract_entities(article.title, article.body)
        events = extract_events(article.title, article.body, entities)
        keywords = self.keywords(article.text)
        return article.evolve(entities=entities, events=events, keywords=keywords)

    def stats(self) -> dict[str, Any]:
        return {
            "background_documents": len(self._background),
            "vocabulary": len(self._document_frequency),
        }
