"""Computing features inside a backtest, from the same code the live engine uses.

Without this the simulator can run but no strategy can decide anything: every
strategy declares ``REQUIRES_FEATURES``, and a backtest that supplies none has
every strategy skip on ``*_unavailable`` for the whole run. That produces a clean
result of zero trades, which looks like "the strategy found no opportunity" and
is really "the strategy was never asked a question it could answer".

The reused-code rule
--------------------
This module builds :class:`~kat.features.pipeline.MarketHistory` and calls
:class:`~kat.features.compute.FeatureComputer` -- the same two objects the live
path uses. It deliberately contains no feature arithmetic of its own. A
simulator that recomputed features its own way would be measuring a strategy
that does not exist, and the discrepancy would show up as unexplained live
underperformance months later.

Point-in-time
-------------
Every accessor on the history filters by ``as_of_ms`` minus that source's
publication lag, so a bar that closes at T is not readable until T + 2s and a
ticker update is not readable for a second. On top of that the engine's
:class:`~kat.simulation.engine.LeakageDetector` independently re-checks every
returned vector. The filtering is the mechanism; the detector is the proof.

Bars are built here rather than generated
-----------------------------------------
Momentum features read one-minute bars, so the provider aggregates the book
mid-price series into bars whose ``ts_ms`` is the bar's **close**. Building them
from the same mids the strategy sees keeps the two consistent; taking them from
a separate generated series would let the bar and the book disagree about what
the price was, which no real market does.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import MarketType
from kat.features.compute import (
    EVENT_FEATURE_SET,
    PERP_FEATURE_SET,
    Bar,
    BarSeries,
    FeatureComputer,
    TradePrint,
    TradeSeries,
)
from kat.features.pipeline import (
    BookObservation,
    FundingObservation,
    MarketHistory,
    TickerObservation,
)
from kat.features.spec import FeatureSetSpec, FeatureVector
from kat.marketdata.book import OrderBook
from kat.marketdata.state import funding_rate_per_hour
from kat.observability.logging import get_logger
from kat.simulation.engine import BacktestEngine, EventKind, SimEvent

log = get_logger(__name__)

__all__ = ["SimFeatureProvider", "build_histories"]

#: Bar period for the momentum features. One minute, matching what the exchange
#: publishes and what :data:`kat.features.spec.LAG_CANDLE_MS` is calibrated for.
BAR_PERIOD_MS = 60_000


@dataclass(slots=True)
class _Accumulator:
    """Per-ticker observations, collected before the history is frozen."""

    books: list[BookObservation] = field(default_factory=list)
    tickers: list[TickerObservation] = field(default_factory=list)
    funding: list[FundingObservation] = field(default_factory=list)
    prints: list[TradePrint] = field(default_factory=list)
    settlement_yes: float | None = None
    close_ts_ms: int | None = None


def _bars_from_mids(mids: list[tuple[int, int]], period_ms: int) -> BarSeries:
    """Aggregate a mid series into OHLC bars stamped with their close time.

    The close stamp is the point. A bar covering 12:00:00-12:00:59 describes
    12:00:59; stamping it with the open would understate its availability by a
    whole period and let a strategy read a bar a minute before it existed.
    """
    buckets: dict[int, list[int]] = defaultdict(list)
    for ts_ms, mid_cc in mids:
        buckets[(ts_ms // period_ms) * period_ms].append(mid_cc)

    bars: list[Bar] = []
    for open_ms in sorted(buckets):
        values = buckets[open_ms]
        bars.append(
            Bar(
                # Close time, not open time.
                ts_ms=open_ms + period_ms - 1,
                open_cc=values[0],
                high_cc=max(values),
                low_cc=min(values),
                close_cc=values[-1],
            )
        )
    return BarSeries(bars, period_ms=period_ms)


def build_histories(
    events: list[SimEvent],
    market_type: MarketType,
    *,
    bar_period_ms: int = BAR_PERIOD_MS,
    funding_interval_hours: float = 8.0,
    close_ts_ms: dict[str, int] | None = None,
) -> dict[str, MarketHistory]:
    """One :class:`MarketHistory` per ticker, built from a simulated feed.

    Building from the whole feed is safe because every accessor on the history
    filters by ``as_of_ms``: the object holds the future but cannot hand it out.
    That is the same arrangement the training pipeline uses over stored history.
    """
    accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)
    mids: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for event in events:
        acc = accumulators[event.ticker]
        # Availability, not information time. The feature layer's own lags are
        # applied on top of this, so a book stamped with when we received it is
        # exactly what the live engine would have had.
        ts = event.available_ms

        if event.kind is EventKind.BOOK:
            book = _book_from_payload(event)
            if book is None:
                continue
            acc.books.append(BookObservation(ts_ms=ts, book=book))
            mid = book.mid_cc
            if mid is not None:
                mids[event.ticker].append((ts, mid))

        elif event.kind is EventKind.TRADE:
            acc.prints.append(
                TradePrint(
                    ts_ms=ts,
                    price_cc=int(event.payload["price_cc"]),
                    qty_qc=int(event.payload["qty_qc"]),
                )
            )

        elif event.kind is EventKind.TICKER:
            payload = event.payload
            acc.tickers.append(
                TickerObservation(
                    ts_ms=ts,
                    volume_qc=payload.get("volume_qc"),
                    open_interest_qc=payload.get("open_interest_qc"),
                    mark_price_cc=payload.get("mark_price_cc"),
                    last_trade_cc=payload.get("mark_price_cc"),
                )
            )
            # The payload carries Kalshi's raw per-*interval* rate; this field
            # is per *hour*. The conversion has exactly one definition.
            hourly = funding_rate_per_hour(
                payload.get("funding_rate"), funding_interval_hours
            )
            if hourly is not None:
                acc.funding.append(
                    FundingObservation(
                        ts_ms=ts,
                        rate_hourly=hourly,
                        next_funding_ms=payload.get("next_funding_ms"),
                    )
                )

        elif event.kind is EventKind.SETTLEMENT:
            acc.settlement_yes = 1.0 if event.payload.get("outcome_yes") else 0.0
            acc.close_ts_ms = ts

    histories: dict[str, MarketHistory] = {}
    for ticker, acc in accumulators.items():
        if not acc.books:
            # A market we never saw a book for is not a market we could trade.
            continue
        histories[ticker] = MarketHistory(
            ticker,
            market_type,
            books=acc.books,
            bars=_bars_from_mids(mids[ticker], bar_period_ms),
            trades=TradeSeries(acc.prints),
            tickers=acc.tickers,
            funding=acc.funding,
            # An explicit definition wins over one inferred from a settlement
            # event: the settlement tells us when the market *resolved*, which
            # is later than when it stopped trading.
            close_ts_ms=(close_ts_ms or {}).get(ticker, acc.close_ts_ms),
            settlement_yes=acc.settlement_yes,
        )
    return histories


def _book_from_payload(event: SimEvent) -> OrderBook | None:
    """Rebuild a book exactly as the engine does.

    Calls :meth:`BacktestEngine._build_book` rather than reimplementing it. A
    second implementation of the YES/NO reflection is the one thing this must
    not have: the features and the strategy would then be looking at two
    different books, and the difference would be invisible.
    """
    try:
        return BacktestEngine._build_book(event)
    except Exception as exc:  # one malformed snapshot must not stop the run
        log.warning("simulation.book_rebuild_failed", ticker=event.ticker, error=str(exc))
        return None


class SimFeatureProvider:
    """A :data:`~kat.simulation.engine.FeatureProvider` backed by a feed.

    Constructed from the same events the backtest will replay, then handed to
    :class:`~kat.simulation.engine.BacktestEngine`, which calls it once per
    market per decision time.
    """

    def __init__(
        self,
        events: list[SimEvent],
        market_type: MarketType,
        *,
        spec: FeatureSetSpec | None = None,
        bar_period_ms: int = BAR_PERIOD_MS,
        funding_interval_hours: float = 8.0,
        close_ts_ms: dict[str, int] | None = None,
        cache: bool = True,
    ) -> None:
        self.market_type = market_type
        self.spec = spec or (
            EVENT_FEATURE_SET if market_type is MarketType.EVENT else PERP_FEATURE_SET
        )
        self.computer = FeatureComputer(self.spec)
        self.histories = build_histories(
            events, market_type,
            bar_period_ms=bar_period_ms,
            funding_interval_hours=funding_interval_hours,
            close_ts_ms=close_ts_ms,
        )
        self._cache: dict[tuple[str, int], FeatureVector | None] = {}
        self._caching = cache
        self.calls = 0
        self.computed = 0
        self.no_history = 0
        self.no_inputs = 0
        self.failures = 0

    def __call__(self, ticker: str, now_ms: int) -> FeatureVector | None:
        self.calls += 1
        key = (ticker, now_ms)
        if self._caching and key in self._cache:
            return self._cache[key]

        vector = self._compute(ticker, now_ms)
        if self._caching:
            self._cache[key] = vector
        return vector

    def _compute(self, ticker: str, now_ms: int) -> FeatureVector | None:
        history = self.histories.get(ticker)
        if history is None:
            self.no_history += 1
            return None

        inputs = history.inputs_at(now_ms)
        if inputs is None:
            # Nothing publishable yet. Returning None is correct and common
            # during warm-up; a zero-filled vector would be a fabrication.
            self.no_inputs += 1
            return None

        try:
            # strict=True re-checks point-in-time inside the computer. Left on
            # in simulation for the same reason it is on live: the check costs
            # microseconds and catches a strategy trading on data it never had.
            vector = self.computer.compute(inputs, as_of_ms=now_ms, strict=True)
        except Exception as exc:
            self.failures += 1
            log.warning(
                "simulation.feature_compute_failed",
                ticker=ticker, now_ms=now_ms, error=str(exc),
            )
            return None

        self.computed += 1
        return vector

    def stats(self) -> dict[str, Any]:
        return {
            "feature_set_id": self.spec.feature_set_id,
            "markets": len(self.histories),
            "calls": self.calls,
            "computed": self.computed,
            "no_history": self.no_history,
            "no_inputs": self.no_inputs,
            "failures": self.failures,
        }
