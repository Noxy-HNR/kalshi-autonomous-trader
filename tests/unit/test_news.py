"""Tests for the news pipeline.

Most of these encode a specific defect that the pipeline actually exhibited the
first time it was pointed at the live exchange and the live feeds. Each one names
the failure in its docstring, because a test called ``test_threshold`` tells a
future reader nothing about why the number is what it is.
"""

from __future__ import annotations

import pytest

from kat.common.enums import MarketType
from kat.common.timeutil import SimulatedClock
from kat.news.dedupe import Deduplicator
from kat.news.extract import Extractor, extract_entities, extract_events
from kat.news.impact import MAX_IMPACT, ImpactEstimator, freshness
from kat.news.matching import (
    DEFAULT_RELEVANCE_THRESHOLD,
    MarketMatcher,
    MarketProfile,
    _breadth_penalty,
)
from kat.news.normalise import (
    clean_text,
    content_hash,
    normalise,
    simhash,
    simhash_distance,
    tokenise,
)
from kat.news.pipeline import _interleave
from kat.news.types import DataQuality, EventType, RawItem

HOUR_MS = 3_600_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def raw(
    title: str,
    body: str = "",
    *,
    source: str = "src_a",
    published_ms: int | None = 1_000_000,
    reliability: float = 0.8,
    external_id: str | None = None,
) -> RawItem:
    return RawItem(
        source_id=source,
        source_name=source.upper(),
        external_id=external_id or f"{source}:{title}",
        title=title,
        summary="",
        body=body,
        url=f"https://example.invalid/{abs(hash(title))}",
        published_ms=published_ms,
        reliability=reliability,
    )


def clock(now_ms: int = 2_000_000) -> SimulatedClock:
    return SimulatedClock(start_ms=now_ms)


def article(title: str, body: str = "", *, now_ms: int = 2_000_000, **kw):
    result = normalise(raw(title, body, **kw), clock(now_ms))
    assert result is not None
    return result


def market(
    ticker: str,
    title: str,
    *,
    rules: str = "",
    event_ticker: str = "",
    close_ms: int | None = None,
) -> MarketProfile:
    return MarketProfile.build(
        ticker=ticker,
        market_type=MarketType.EVENT,
        title=title,
        event_ticker=event_ticker,
        rules=rules,
        close_ms=close_ms,
    )


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_ingested_time_comes_from_the_clock_not_the_source(self):
        """The point-in-time guarantee rests entirely on this.

        ``published_ms`` is whatever the source claims. ``ingested_ms`` is when
        this process could first have acted. A backtest filters on the second,
        so it must never be derived from the first.
        """
        a = article("Fed holds rates", now_ms=5_000_000, published_ms=1_000)
        assert a.ingested_ms == 5_000_000
        assert a.published_ms == 1_000

    def test_a_back_dated_item_is_still_fresh_to_us(self):
        """A feed that publishes six-hour-old items has not published stale news.

        We could not have acted on it before we received it, so its age is
        measured from ingestion. Getting this backwards would silently discard
        every government feed that batches its releases.
        """
        a = article("Treasury publishes final rule on reporting",
                    now_ms=10 * HOUR_MS, published_ms=1)
        assert a.age_ms(10 * HOUR_MS) == 0
        assert freshness(a, 10 * HOUR_MS) == pytest.approx(1.0)

    def test_missing_publish_time_is_flagged_not_invented(self):
        a = article("Congress passes appropriations measure", published_ms=None)
        assert a.published_ms is None
        assert a.quality is DataQuality.PARTIAL

    def test_future_dated_item_is_suspect(self):
        a = article("Senate confirms nominee to federal bench",
                    now_ms=1_000, published_ms=99_000_000)
        assert a.quality is DataQuality.SUSPECT

    def test_empty_title_is_dropped(self):
        assert normalise(raw("   "), clock()) is None

    def test_html_is_stripped(self):
        text = clean_text("<p>Fed <b>holds</b> rates</p>&amp; more")
        assert "<" not in text
        assert "Fed holds rates" in text
        assert "&" in text

    def test_content_hash_ignores_formatting_but_not_words(self):
        assert content_hash("Fed holds", "a b") == content_hash("  Fed   holds ", "a  b")
        assert content_hash("Fed holds", "a b") != content_hash("Fed cuts", "a b")


class TestSimhash:
    def test_near_identical_texts_are_close(self):
        a = simhash("The Federal Reserve held interest rates steady on Wednesday")
        b = simhash("The Federal Reserve held interest rates steady Wednesday")
        assert simhash_distance(a, b) <= 6

    def test_unrelated_texts_are_far(self):
        a = simhash("The Federal Reserve held interest rates steady on Wednesday")
        b = simhash("Hurricane makes landfall on the Gulf Coast overnight")
        assert simhash_distance(a, b) > 12

    def test_distance_to_self_is_zero(self):
        h = simhash("anything at all goes here")
        assert simhash_distance(h, h) == 0

    def test_empty_text_has_no_fingerprint(self):
        assert simhash("") == 0


# ---------------------------------------------------------------------------
# deduplication
# ---------------------------------------------------------------------------


class TestDeduplicator:
    def test_identical_text_from_two_outlets_collapses_to_one(self):
        d = Deduplicator()
        body = "The Federal Reserve held interest rates steady, citing inflation."
        kept = d.process(
            [
                article("Fed holds rates", body, source="wire"),
                article("Fed holds rates", body, source="reprint"),
            ]
        )
        assert len(kept) == 1
        assert d.exact_duplicates == 1

    def test_a_reprint_counts_as_corroboration(self):
        """The whole reason to deduplicate rather than drop.

        Twelve outlets carrying one wire story is evidence the story is real.
        Counting it as twelve articles would turn "independently confirmed" into
        "how many outlets subscribe to the same wire".
        """
        d = Deduplicator()
        body = "Officials confirmed the ceasefire agreement early on Tuesday."
        kept = d.process(
            [
                article("Ceasefire agreed", body, source="wire"),
                article("Ceasefire agreed", body, source="outlet_b"),
                article("Ceasefire agreed", body, source="outlet_c"),
            ]
        )
        assert len(kept) == 1
        assert kept[0].corroborations == 2

    def test_the_same_outlet_repeating_itself_is_not_corroboration(self):
        d = Deduplicator()
        body = "Officials confirmed the ceasefire agreement early on Tuesday."
        kept = d.process(
            [
                article("Ceasefire agreed", body, source="wire", external_id="1"),
                article("Ceasefire agreed", body, source="wire", external_id="2"),
            ]
        )
        assert len(kept) == 1
        assert kept[0].corroborations == 0

    def test_different_stories_are_not_merged(self):
        d = Deduplicator()
        kept = d.process(
            [
                article("Fed holds rates steady", "The central bank left policy unchanged."),
                article("Hurricane hits Florida", "The storm made landfall near Tampa."),
            ]
        )
        assert len(kept) == 2

    def test_a_repeat_outside_the_window_is_a_new_event(self):
        """Two "Fed holds rates" a month apart are two decisions, not a reprint."""
        d = Deduplicator(window_ms=HOUR_MS)
        body = "The central bank left policy unchanged."
        first = article("Fed holds rates", body, now_ms=1_000_000)
        later = article("Fed holds rates", body, now_ms=1_000_000 + 5 * HOUR_MS, source="src_b")
        assert d.process([first]) != []
        assert d.process([later]) != []
        assert d.exact_duplicates == 0


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_gazetteer_entities_are_found(self):
        entities = extract_entities("Federal Reserve holds rates", "")
        kinds = {e.normalised for e in entities}
        assert "federal_reserve" in kinds

    def test_headline_entities_outrank_body_entities(self):
        entities = extract_entities(
            "Kaja Kallas named to post",
            "A spokesperson for Klaus Iohannis declined to comment.",
        )
        by_name = {e.normalised: e.salience for e in entities}
        assert by_name["kaja_kallas"] > by_name["klaus_iohannis"]

    def test_typographic_apostrophes_do_not_split_a_name(self):
        """Real headlines use U+2019, and a name split in half is a lost name."""
        entities = extract_entities(f"Trump{chr(0x2019)}s advisor resigns", "")
        assert any("trump" in e.normalised for e in entities)

    def test_event_polarity_is_relative_to_its_subject(self):
        entities = extract_entities("Smith wins the primary", "")
        events = extract_events("Smith wins the primary", "", entities)
        assert events
        win = next(e for e in events if e.event_type is EventType.ELECTION_RESULT)
        assert win.polarity == 1

    def test_keywords_come_only_from_the_articles_own_text(self):
        """Guards against cross-contamination through the shared background corpus."""
        ex = Extractor()
        ex.enrich(article("Best high-yield savings rates today", "APY and interest and yield."))
        heat = ex.enrich(article("Extreme Heat Watch issued", "Temperatures will reach 110 F."))
        own = set(tokenise(heat.text))
        assert all(term in own for term, _ in heat.keywords)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


class TestMatching:
    def test_a_relevant_story_matches_its_market(self):
        m = MarketMatcher()
        m.set_markets([market("FED-1", "Will the Fed cut rates in September?",
                              rules="Federal Reserve target range for the federal funds rate.")])
        ex = Extractor()
        a = ex.enrich(article(
            "Federal Reserve signals a September rate cut",
            "The Federal Reserve indicated the federal funds rate could fall.",
        ))
        matches = m.match(a)
        assert matches and matches[0].ticker == "FED-1"

    def test_an_unrelated_story_matches_nothing(self):
        m = MarketMatcher()
        m.set_markets([market("FED-1", "Will the Fed cut rates in September?")])
        ex = Extractor()
        a = ex.enrich(article("Hurricane makes landfall near Tampa",
                              "The storm brought heavy rain to the Gulf Coast."))
        assert m.match(a) == []

    def test_resolution_boilerplate_does_not_score(self):
        """The defect: every market's rules end "...then the market resolves to Yes".

        Those tokens are shared by the whole exchange, so an article containing
        the word "market" or "yes" earned keyword credit against every market
        listed. Live, this scored "Yes, yields are important, but don't forget
        about Nvidia" at 0.70 against a 2036 fed funds strike.
        """
        p = market(
            "FED-1",
            "Will the upper bound of the federal funds rate be above 6.00%?",
            rules="If the rate is above 6.00%, then the market resolves to Yes.",
        )
        for boilerplate in ("market", "resolves", "yes", "then", "above"):
            assert boilerplate not in p.tokens
        for real_content in ("federal", "funds", "rate", "bound", "upper"):
            assert real_content in p.tokens

    def test_a_short_entity_does_not_match_inside_a_longer_word(self):
        """The defect: raw substring matching made "us" match "because"/"trust".

        Live, that scored "76 percent in Canada say ending US trade negotiations
        was right" against "US nominal GDP growth in 2036" on the lone entity
        `us` and the lone keyword `percent`.
        """
        m = MarketMatcher(threshold=0.0)
        p = market("X-1", "Will the industry trust because of this?")
        ex = Extractor()
        a = ex.enrich(article("US officials comment", "US trade policy shifted."))
        scored = m.score(a, p)
        assert scored is None or "us" not in scored.entity_overlap

    def test_a_broad_market_title_is_penalised(self):
        """The defect: Kalshi parlay markets list every leg in one title.

        With overlap normalised only against the article, those markets matched
        essentially every article and crowded out the markets a story was
        genuinely about. That is exactly what happened on the first live run.
        """
        assert _breadth_penalty(50) == 1.0
        assert _breadth_penalty(90) == 1.0
        assert _breadth_penalty(360) == pytest.approx(0.5)
        assert _breadth_penalty(900) < _breadth_penalty(360)

    def test_the_threshold_rejects_a_lone_headline_entity(self):
        """One headline entity with no keyword corroboration scores 0.367.

        Measured on live data: 5,728 of 20,578 non-zero pairs landed on exactly
        that value. The threshold sits above it deliberately.
        """
        assert DEFAULT_RELEVANCE_THRESHOLD > 0.367

    def test_a_closed_market_cannot_be_traded_on_new_news(self):
        m = MarketMatcher()
        m.set_markets([market("FED-1", "Will the Fed cut rates?",
                              rules="Federal Reserve federal funds rate.", close_ms=1_000)])
        ex = Extractor()
        a = ex.enrich(article("Federal Reserve cuts the federal funds rate",
                              "The Federal Reserve cut rates.", now_ms=2_000_000))
        assert m.match(a) == []

    def test_a_strike_ladder_counts_as_one_event(self):
        """The defect: a per-market cap spent its whole budget on one ladder.

        A single Fed story legitimately bears on every rung of a rate ladder, so
        capping at five markets kept five rungs of one event and dropped every
        other event entirely. The cap is on events; ladders stay intact.
        """
        m = MarketMatcher(max_events_per_article=2, max_matches_per_article=50)
        rungs = [
            market(f"FED-T{i}", "Fed funds rate at end of 2036",
                   rules="Federal Reserve target range for the federal funds rate.",
                   event_ticker="FEDEVENT")
            for i in range(8)
        ]
        other = market("NATO-1", "Who will be the next NATO Secretary General?",
                       rules="NATO Secretary General appointment.", event_ticker="NATOEVENT")
        m.set_markets([*rungs, other])
        ex = Extractor()
        a = ex.enrich(article(
            "Federal Reserve holds the federal funds rate",
            "The Federal Reserve left its target range unchanged.",
        ))
        matches = m.match(a)
        assert len({x.event_ticker for x in matches}) <= 2
        assert len([x for x in matches if x.event_ticker == "FEDEVENT"]) == 8


# ---------------------------------------------------------------------------
# impact
# ---------------------------------------------------------------------------


class TestImpact:
    def _estimate(self, headline: str, market_title: str, *, body: str = "", relevance=0.8):
        ex = Extractor()
        a = ex.enrich(article(headline, body))
        return ImpactEstimator().estimate(
            a, ticker="T-1", market_title=market_title,
            relevance=relevance, now_ms=a.ingested_ms,
        )

    def test_no_direction_when_the_market_does_not_name_the_subject(self):
        """"Smith wins primary" is +1 for Smith and says nothing about Jones.

        Nothing in the sentence reveals which side of a Jones market Smith sits
        on, so the honest answer is no view.
        """
        est = self._estimate("Smith wins the primary", "Will Jones win the nomination?")
        assert est.direction is None
        assert not est.has_view

    def test_direction_when_the_market_does_name_the_subject(self):
        est = self._estimate("Smith wins the primary", "Will Smith win the nomination?")
        assert est.direction == "yes"

    def test_an_article_with_no_typed_event_yields_no_view(self):
        est = self._estimate("Analysts discuss the outlook", "Will the Fed cut rates?")
        assert est.direction is None
        assert est.magnitude == 0.0

    def test_impact_is_capped(self):
        est = self._estimate("Smith wins the election", "Will Smith win the election?")
        assert 0.0 <= est.magnitude <= MAX_IMPACT

    def test_impact_decays_and_then_stops(self):
        a = Extractor().enrich(article("Smith wins the election", now_ms=0))
        assert freshness(a, 0) == pytest.approx(1.0)
        assert freshness(a, 45 * 60_000) == pytest.approx(0.5)
        assert freshness(a, 7 * HOUR_MS) == 0.0

    def test_a_stale_article_carries_no_view(self):
        ex = Extractor()
        a = ex.enrich(article("Smith wins the election", now_ms=0))
        est = ImpactEstimator().estimate(
            a, ticker="T-1", market_title="Will Smith win the election?",
            relevance=0.9, now_ms=7 * HOUR_MS,
        )
        assert est.magnitude == 0.0
        assert est.direction is None

    def test_confidence_below_the_floor_removes_the_direction(self):
        est = self._estimate(
            "Smith wins the primary", "Will Smith win the nomination?", relevance=0.01
        )
        if est.confidence < ImpactEstimator().min_confidence:
            assert est.direction is None


# ---------------------------------------------------------------------------
# pipeline plumbing
# ---------------------------------------------------------------------------


class TestInterleave:
    def test_the_cap_is_spent_evenly_across_sources(self):
        """The defect: concatenate-then-truncate starved the last sources forever.

        With twenty sources and a binding cap, the sources at the end of the list
        were cut on every single cycle and were effectively never ingested.
        """
        batches = [list("abcde"), list("AB"), list("12345678")]
        assert "".join(_interleave(batches, 6)) == "aA1bB2"

    def test_nothing_is_lost_when_the_cap_does_not_bind(self):
        batches = [list("abcde"), list("AB"), list("12345678")]
        assert sorted(_interleave(batches, 100)) == sorted("abcdeAB12345678")

    def test_degenerate_inputs(self):
        assert _interleave([], 5) == []
        assert _interleave([[1, 2]], 0) == []


class TestFuturePublicationTimes:
    """A story cannot be published after we receive it.

    The case that motivated the tolerance: ESPN stamps its August items ``EST``
    when US Eastern is on ``EDT``, placing every one of them 59 minutes in the
    future. A one-hour threshold missed all of them by a minute.
    """

    def test_an_hour_ahead_is_flagged(self):
        a = article(
            "Browns name their starting quarterback for the season",
            now_ms=10_000_000,
            published_ms=10_000_000 + 59 * 60_000,
        )
        assert a.quality is DataQuality.SUSPECT

    def test_ordinary_clock_skew_is_tolerated(self):
        a = article(
            "Senate confirms nominee to the federal bench",
            now_ms=10_000_000,
            published_ms=10_000_000 + 5_000,
        )
        assert a.quality is DataQuality.OK

    def test_a_lying_publication_time_cannot_affect_freshness(self):
        """The safety property: nothing downstream reads published_ms.

        Freshness, decay and backtest filtering all measure from ingestion, so a
        source an hour ahead of itself changes a data-quality flag and nothing
        that touches a trading decision.
        """
        honest = article("Fed holds rates steady this month", now_ms=5_000_000,
                         published_ms=5_000_000 - 60_000)
        lying = article("Fed holds rates steady this month", now_ms=5_000_000,
                        published_ms=5_000_000 + 59 * 60_000, source="src_b")
        assert honest.age_ms(6_000_000) == lying.age_ms(6_000_000)
        assert freshness(honest, 6_000_000) == freshness(lying, 6_000_000)


class TestNumbersAreContent:
    """A headline's number is often the entire story.

    The defect: `content_hash` hashed the output of `tokenise`, which drops
    stopwords and any token shorter than two characters. Single digits vanished
    and decimals were split, so "Hurricane reaches Category 3" and "Category 5"
    produced identical hashes and the second was silently discarded as a
    duplicate of the first. Found by noticing that 6,000 articles with distinct
    titles were collapsing into a handful.
    """

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("Fed cuts rates to 4.25%", "Fed cuts rates to 4.50%"),
            ("CPI rose 3.1% in July", "CPI rose 2.4% in July"),
            ("Hurricane reaches Category 3", "Hurricane reaches Category 5"),
            ("Trump leads by 5 points", "Trump leads by 12 points"),
            ("Payrolls up 150000", "Payrolls up 90000"),
            ("Decision number 1", "Decision number 2"),
        ],
    )
    def test_a_different_number_is_a_different_story(self, first, second):
        assert content_hash(first, "") != content_hash(second, "")

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("Fed Holds Rates!", "fed holds rates"),
            ("A  b   c", "a b c"),
            ("x, y; z", "x y z"),
            ("Rates steady -- again", "rates steady again"),
        ],
    )
    def test_formatting_alone_is_still_the_same_story(self, first, second):
        """The behaviour the token-based hash was reaching for, kept."""
        assert content_hash(first, "") == content_hash(second, "")

    def test_a_decimal_survives_tokenisation_whole(self):
        """Splitting 4.25 into 4 and 25 loses the number entirely."""
        assert "4.25" in tokenise("Fed cuts to 4.25% today")

    def test_lone_digits_survive_but_lone_letters_do_not(self):
        tokens = tokenise("Category 3 storm x y z")
        assert "3" in tokens
        assert "x" not in tokens and "y" not in tokens

    def test_numerically_different_headlines_are_not_near_duplicates(self):
        """Exact hashing is not enough: simhash must separate them too.

        With digits dropped, these two tokenised identically -- so even with
        different content hashes the near-duplicate check would have merged them.
        """
        pairs = [
            ("Hurricane reaches Category 3", "Hurricane reaches Category 5"),
            ("CPI rose 3.1% in July", "CPI rose 2.4% in July"),
        ]
        for a, b in pairs:
            assert simhash_distance(simhash(a), simhash(b)) > 3, (a, b)

    def test_the_deduplicator_keeps_both(self):
        """End to end: the story that used to disappear."""
        d = Deduplicator()
        kept = d.process(
            [
                article("Hurricane reaches Category 3", "The storm intensified overnight."),
                article("Hurricane reaches Category 5", "The storm intensified overnight.",
                        source="src_b"),
            ]
        )
        assert len(kept) == 2, "a stronger hurricane is not a reprint of a weaker one"
