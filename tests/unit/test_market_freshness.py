"""Freshness must distinguish a quiet market from a dead feed.

This is a delta feed: the exchange sends a message when the book *changes*, so a
market that has sent nothing is usually reporting that our copy is still exact.
The old test measured every market against a five-second budget, which on Kalshi
meant nine cycles in ten declined every market being watched -- a platform that
could not trade at all.

The dangerous silence is silence across the whole feed, because that is
indistinguishable from disconnection. These tests pin both halves: the quiet
market stays tradeable, and everything goes stale the moment the feed does.
"""

from __future__ import annotations

import pytest

from kat.common.enums import MarketType
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState

BUDGET = 5_000
QUIET = 300_000
T0 = 1_000_000


def _state() -> MarketState:
    return MarketState(MarketType.EVENT, max_age_ms=BUDGET, quiet_max_age_ms=QUIET)


def _tick(state: MarketState, ticker: str, now_ms: int) -> None:
    """Apply a real message, which is what stamps both clocks."""
    state.apply_snapshot(ticker, OrderBook(ticker=ticker), now_ms=now_ms)


def test_never_seen_market_is_not_fresh() -> None:
    """A book we have never seen is a definite refusal, not an unknown."""
    assert _state().is_fresh("NOPE", now_ms=T0) is False


def test_recently_updated_market_is_fresh() -> None:
    state = _state()
    _tick(state, "A", T0)
    assert state.is_fresh("A", now_ms=T0 + 1_000) is True


def test_quiet_market_stays_fresh_while_the_feed_is_live() -> None:
    """The case that made the platform untradeable: 7.5s between updates."""
    state = _state()
    _tick(state, "QUIET", T0)
    # Another market keeps proving the connection is delivering.
    _tick(state, "BUSY", T0 + 7_000)
    assert state.is_fresh("QUIET", now_ms=T0 + 7_500) is True


def test_quiet_market_goes_stale_when_the_whole_feed_goes_quiet() -> None:
    """The failure mode that matters is preserved exactly."""
    state = _state()
    _tick(state, "A", T0)
    _tick(state, "B", T0)
    # Nothing at all for longer than the budget: we may simply be disconnected.
    assert state.is_fresh("A", now_ms=T0 + BUDGET + 1) is False
    assert state.is_fresh("B", now_ms=T0 + BUDGET + 1) is False


def test_a_disconnect_stales_every_market_at_once() -> None:
    state = _state()
    for ticker in ("A", "B", "C"):
        _tick(state, ticker, T0)
    later = T0 + BUDGET + 1
    assert state.stale_tickers(now_ms=later) == ["A", "B", "C"]


def test_quiet_market_goes_stale_past_the_quiet_bound_even_on_a_live_feed() -> None:
    """A live feed does not prove *this* subscription is still live."""
    state = _state()
    _tick(state, "DROPPED", T0)
    now = T0 + QUIET + 1
    _tick(state, "BUSY", now)  # feed demonstrably alive
    assert state.is_fresh("DROPPED", now_ms=now) is False


def test_a_stricter_caller_budget_still_admits_a_quiet_market_on_a_live_feed() -> None:
    """Strategies pass their own budget; the two-part test still applies."""
    state = _state()
    _tick(state, "QUIET", T0)
    _tick(state, "BUSY", T0 + 900)
    assert state.is_fresh("QUIET", now_ms=T0 + 1_000, max_age_ms=1_000) is True


def test_a_stricter_caller_budget_narrows_what_counts_as_a_live_feed() -> None:
    """A caller demanding 1s freshness also demands 1s of proof of liveness."""
    state = _state()
    _tick(state, "QUIET", T0)
    _tick(state, "BUSY", T0 + 1_000)
    # Feed last spoke 2s ago, which is outside this caller's 1s budget.
    assert state.is_fresh("QUIET", now_ms=T0 + 3_000, max_age_ms=1_000) is False


def test_feed_age_is_unknown_before_the_first_message() -> None:
    """None, not zero: an unknown feed age is not a proven-live one."""
    assert _state().feed_age_ms(now_ms=T0) is None


def test_feed_age_tracks_the_most_recent_message_on_any_ticker() -> None:
    state = _state()
    _tick(state, "A", T0)
    _tick(state, "B", T0 + 2_000)
    assert state.feed_age_ms(now_ms=T0 + 2_500) == 500


@pytest.mark.parametrize("gap_ms", [0, 1, BUDGET - 1, BUDGET])
def test_market_within_its_own_budget_never_consults_the_feed(gap_ms: int) -> None:
    """The fast path must not depend on another market having reported."""
    state = _state()
    _tick(state, "ONLY", T0)
    assert state.is_fresh("ONLY", now_ms=T0 + gap_ms) is True


def test_stats_expose_the_feed_liveness_used_for_the_decision() -> None:
    """An operator asking "why is nothing trading" must be able to see this."""
    state = _state()
    _tick(state, "A", T0)
    stats = state.stats(now_ms=T0 + 1_500)
    assert stats["feed_age_ms"] == 1_500
    assert stats["quiet_max_age_ms"] == QUIET


# -- the strategy-level screen must agree with the loop ----------------------
#
# Freshness was checked in three places: the loop's gate, the loop's
# per-strategy gate, and each strategy's own screen. The first two learned that
# a quiet market on a live feed is actable; the third did not, so strategies
# went on declining markets the loop had just admitted, and stale_market_data
# stayed the single largest skip reason. These pin the two halves together.

from kat.marketdata.state import MarketSnapshot  # noqa: E402
from kat.strategies.base import StrategyContext  # noqa: E402
from kat.strategies.proposal import SKIP_NO_TIMESTAMP, SKIP_STALE, screen_freshness  # noqa: E402


def _ctx(*, age_ms: int, feed_live: bool | None) -> StrategyContext:
    snap = MarketSnapshot(ticker="X", market_type=MarketType.EVENT)
    snap.updated_ms = T0
    return StrategyContext(now_ms=T0 + age_ms, market=snap, feed_live=feed_live)


def test_strategy_screen_accepts_a_fresh_market() -> None:
    assert screen_freshness(_ctx(age_ms=1_000, feed_live=True), BUDGET) is None


def test_strategy_screen_accepts_a_quiet_market_on_a_live_feed() -> None:
    """The case that kept stale_market_data as the top skip reason."""
    assert screen_freshness(_ctx(age_ms=BUDGET * 3, feed_live=True), BUDGET) is None


def test_strategy_screen_rejects_a_stale_market_on_a_dead_feed() -> None:
    assert screen_freshness(_ctx(age_ms=BUDGET * 3, feed_live=False), BUDGET) == SKIP_STALE


def test_strategy_screen_keeps_the_strict_rule_when_liveness_is_unknown() -> None:
    """Backtests build contexts by hand and must not silently loosen."""
    assert screen_freshness(_ctx(age_ms=BUDGET * 3, feed_live=None), BUDGET) == SKIP_STALE


def test_strategy_screen_still_refuses_an_unstamped_snapshot() -> None:
    """A live feed does not excuse a market we have never received."""
    snap = MarketSnapshot(ticker="X", market_type=MarketType.EVENT)
    ctx = StrategyContext(now_ms=T0, market=snap, feed_live=True)
    assert screen_freshness(ctx, BUDGET) == SKIP_NO_TIMESTAMP
