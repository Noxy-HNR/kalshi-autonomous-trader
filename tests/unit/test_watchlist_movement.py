"""The watchlist must be able to tell a still market from a busy one.

A REST listing ranks markets by what they have already done -- 24-hour volume,
current spread, time to close. None of that says whether a market will move in
the next half hour, and on Kalshi the highest-volume contracts are often
long-dated ones that reprice a few times a day. Watching sixty of those yields
neither training data nor a trading opportunity.

The dedup in `_persist_quote` already decides "did the top of book actually
change" on every message. These tests pin that the answer is kept per ticker and
consumed by the refresh, because that is the platform's only first-hand evidence
about its own subscriptions.
"""

from __future__ import annotations

from kat.common.enums import MarketType
from kat.common.timeutil import SystemClock
from kat.engine.ingestion import MarketDataIngestor
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState


class _NullWS:
    connected = True

    def on(self, *_a: object, **_k: object) -> None:
        pass


class _CollectingWriter:
    def __init__(self) -> None:
        self.rows: list[object] = []

    def enqueue(self, obj: object, **_k: object) -> bool:
        self.rows.append(obj)
        return True


def _ingestor() -> MarketDataIngestor:
    return MarketDataIngestor(
        market_type=MarketType.EVENT,
        ws=_NullWS(),  # type: ignore[arg-type]
        state=MarketState(MarketType.EVENT),
        clock=SystemClock(),
        writer=_CollectingWriter(),  # type: ignore[arg-type]
        persist_quotes=True,
    )


def _apply(ing: MarketDataIngestor, ticker: str, bid: int, ask: int, ts: int) -> None:
    book = OrderBook.from_event_contract_levels(
        ticker, yes_levels=[(bid, 1000)], no_levels=[(10_000 - ask, 1000)]
    )
    ing.state.apply_snapshot(ticker, book, now_ms=ts)
    ing._persist_quote(ticker, ts)


def test_a_market_that_moves_is_counted() -> None:
    ing = _ingestor()
    _apply(ing, "MOVER", 3000, 3100, 1_000)
    _apply(ing, "MOVER", 3100, 3200, 2_000)
    _apply(ing, "MOVER", 3200, 3300, 3_000)
    assert ing.quote_changes["MOVER"] == 3


def test_a_market_that_reprints_the_same_quote_is_not() -> None:
    """The whole point: identical top-of-book is not movement.

    This is the case that filled the watchlist -- a market can deliver messages
    all day and never change its quote.
    """
    ing = _ingestor()
    for ts in range(1_000, 6_000, 1_000):
        _apply(ing, "STILL", 3000, 3100, ts)
    assert ing.quote_changes["STILL"] == 1  # only the first is a change
    assert ing.quotes_unchanged == 4


def test_movement_is_consumed_not_accumulated() -> None:
    """Each refresh judges the interval it just lived through.

    A cumulative count would let a market that was busy this morning hold its
    slot all evening, which is the behaviour being fixed.
    """
    ing = _ingestor()
    _apply(ing, "A", 3000, 3100, 1_000)
    _apply(ing, "A", 3100, 3200, 2_000)

    first = ing.take_movement()
    assert first == {"A": 2}
    assert ing.take_movement() == {}, "a second read must not see the first interval"

    _apply(ing, "A", 3200, 3300, 3_000)
    assert ing.take_movement() == {"A": 1}


def test_a_market_never_seen_is_absent_rather_than_zero() -> None:
    """The refresh reads this with .get(ticker, 0), so absence must mean no
    movement -- not raise, and not be confused with a market that was watched
    and stayed still."""
    ing = _ingestor()
    movement = ing.take_movement()
    assert movement.get("NEVER-WATCHED", 0) == 0


def test_movement_distinguishes_the_two_populations() -> None:
    """The decision the refresh actually makes."""
    ing = _ingestor()
    for ts in range(1_000, 5_000, 1_000):
        _apply(ing, "STILL", 5000, 5100, ts)
        _apply(ing, "MOVER", 5000 + ts // 100, 5100 + ts // 100, ts)

    movement = ing.take_movement()
    stagnant = [t for t in ("STILL", "MOVER") if movement.get(t, 0) == 0]
    assert stagnant == [], "both moved at least once"
    assert movement["MOVER"] > movement["STILL"]
