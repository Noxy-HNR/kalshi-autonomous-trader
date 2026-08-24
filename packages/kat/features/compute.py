"""Computing feature values, with one rule that shapes everything else.

**Every feature is computed from exactly one source.**

That rule is not aesthetic. :mod:`kat.features.spec` gives each source a single
availability lag and derives ``available_ms = info_ts_ms + lag``. A feature that
blends an order book (lag 0) with a closed candle (lag 2000ms) has no honest
single lag: charging the candle lag against the book's information timestamp
back-dates nothing and instead pushes ``available_ms`` past the decision time, so
:meth:`FeatureVector.require_point_in_time` would reject every live vector;
charging the book lag understates when the value could really be held, which is
the failure the timestamps exist to catch. Rather than weaken the check for the
convenience of a handful of features, blended features simply do not exist here.
Where a book-and-candle comparison is genuinely wanted, it is expressed as two
separate features and the model is left to combine them -- which is what models
are for.

The one apparent exception is the platform clock. ``time_to_expiry`` reads
``as_of_ms``, and features like ``expiry_scaled_move_cc`` multiply a candle
statistic by a clock quantity. A clock reading is not an observation: it is known
arbitrarily far in advance and carries no publication delay, so it contributes no
information timestamp. Such features declare the candle as their source and carry
the candle's timestamp, which is exactly right.

Why these features
------------------
Prediction markets are not equities with a dollar sign. Three consequences run
through the whole catalogue:

* **Price moves are additive, not multiplicative.** A contract going from $0.05
  to $0.10 has doubled, but what happened is that an implied probability rose by
  five points. Every momentum and volatility feature is therefore denominated in
  centicents of price change, never in percent, so that a five-point move ranks
  the same at $0.05 and at $0.50.
* **Level matters on its own.** A contract at $0.97 behaves nothing like one at
  $0.50: it has bounded upside, asymmetric tails, and a different microstructure.
  ``price_level`` and ``distance_from_half`` are real features, not normalisation
  artefacts.
* **The clock is a state variable.** An event contract converges to 0 or 1 at a
  known instant. Volatility that is irrelevant with a month to run is decisive
  with an hour, which is what ``expiry_scaled_move_cc`` expresses.

Perpetuals get carry features instead of term features -- they never expire, so
time to expiry is undefined for them, and funding is the cost that replaces it.

Costs are features too
----------------------
``effective_spread_cc`` walks the book for a reference size instead of reading
the touch. A model trained on touch spreads learns that trading is cheap and is
systematically wrong about exactly the markets it is most tempted to trade: thin
ones. Its ``fill_value`` is the pessimistic extreme rather than zero, so a
missing cost reads as "expensive" and never as "free".

Integers and floats
-------------------
Prices and sizes arrive as integers at the platform scales and stay integers
through every difference and sum. Floats appear only at the end, where a feature
is a ratio, a standard deviation or a logarithm -- statistics with no exact
decimal form. No money value is ever produced by this module.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from kat.common.enums import MarketType
from kat.common.errors import DataQualityError
from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT
from kat.features.spec import (
    LAG_BOOK_MS,
    LAG_CANDLE_MS,
    LAG_CLOCK_MS,
    LAG_FUNDING_MS,
    LAG_TICKER_MS,
    FeatureFamily,
    FeatureSetSpec,
    FeatureSource,
    FeatureSpec,
    FeatureValue,
    FeatureVector,
)
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot
from kat.observability.metrics import metrics
from kat.strategies.common import (
    F_BOOK_IMBALANCE,
    F_MID_CHANGE_1M_CC,
    F_MID_CHANGE_5M_CC,
    F_MID_CHANGE_15M_CC,
    F_REALIZED_VOL_5M_CC,
    F_TIME_TO_EXPIRY_MS,
    F_TRADE_IMBALANCE_5M,
    F_VOLUME_5M_QC,
    F_VOLUME_ZSCORE_5M,
)

__all__ = [
    "ALL_SPECS",
    "EVENT_FEATURE_SET",
    "F_BASIS_CC",
    "F_BOOK_IMBALANCE",
    "F_BOOK_PRESSURE_L1",
    "F_DEPTH_CONCENTRATION",
    "F_DEPTH_TOTAL_LOG",
    "F_DISTANCE_FROM_HALF",
    "F_EFFECTIVE_SPREAD_CC",
    "F_EXPIRY_SCALED_MOVE_CC",
    "F_FUNDING_CARRY_8H_CC",
    "F_FUNDING_RATE_BPS_HOURLY",
    "F_HOURS_TO_FUNDING",
    "F_LOG_TIME_TO_EXPIRY_H",
    "F_MEAN_REVERSION_15M_CC",
    "F_MEAN_REVERSION_60M_CC",
    "F_MICROPRICE_OFFSET_CC",
    "F_MID_CHANGE_1M_CC",
    "F_MID_CHANGE_5M_CC",
    "F_MID_CHANGE_15M_CC",
    "F_MID_CHANGE_30S_CC",
    "F_MID_CHANGE_60M_CC",
    "F_MID_VOL_60S_CC",
    "F_OI_CHANGE_5M_QC",
    "F_OI_CHANGE_RATE_5M",
    "F_PRICE_LEVEL",
    "F_RANGE_POSITION_60M",
    "F_RANGE_VOL_15M_CC",
    "F_RANGE_WIDTH_60M_CC",
    "F_REALIZED_VOL_5M_CC",
    "F_REALIZED_VOL_60M_CC",
    "F_RELATIVE_SPREAD",
    "F_SPREAD_CC",
    "F_TIME_TO_EXPIRY_MS",
    "F_TRADE_IMBALANCE_5M",
    "F_VOLUME_5M_QC",
    "F_VOLUME_ZSCORE_5M",
    "F_VOL_RATIO_15M_60M",
    "PERP_FEATURE_SET",
    "REFERENCE_SIZE_QC",
    "Bar",
    "BarSeries",
    "FeatureComputer",
    "FeatureInputs",
    "MidSeries",
    "TradePrint",
    "TradeSeries",
    "event_feature_set",
    "perp_feature_set",
    "spec_by_name",
]


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
# The nine names re-exported from kat.strategies.common are a contract with the
# strategy layer: a strategy declares them in ``requires_features`` and the
# engine refuses to run it when this module cannot supply them. They are
# imported rather than repeated so the two halves cannot drift.

F_SPREAD_CC = "spread_cc"
F_RELATIVE_SPREAD = "relative_spread"
F_MICROPRICE_OFFSET_CC = "microprice_offset_cc"
F_BOOK_PRESSURE_L1 = "book_pressure_l1"
F_EFFECTIVE_SPREAD_CC = "effective_spread_cc"
F_DEPTH_TOTAL_LOG = "depth_total_log"
F_DEPTH_CONCENTRATION = "depth_concentration"

F_MID_CHANGE_30S_CC = "mid_change_30s_cc"
F_MID_CHANGE_60M_CC = "mid_change_60m_cc"
F_MEAN_REVERSION_15M_CC = "mean_reversion_15m_cc"
F_MEAN_REVERSION_60M_CC = "mean_reversion_60m_cc"

F_MID_VOL_60S_CC = "mid_vol_60s_cc"
F_REALIZED_VOL_60M_CC = "realized_vol_60m_cc"
F_VOL_RATIO_15M_60M = "vol_ratio_15m_60m"
F_RANGE_VOL_15M_CC = "range_vol_15m_cc"

F_OI_CHANGE_5M_QC = "oi_change_5m_qc"
F_OI_CHANGE_RATE_5M = "oi_change_rate_5m"

F_PRICE_LEVEL = "price_level"
F_DISTANCE_FROM_HALF = "distance_from_half"
F_RANGE_POSITION_60M = "range_position_60m"
F_RANGE_WIDTH_60M_CC = "range_width_60m_cc"

F_LOG_TIME_TO_EXPIRY_H = "log_time_to_expiry_h"
F_EXPIRY_SCALED_MOVE_CC = "expiry_scaled_move_cc"

F_FUNDING_RATE_BPS_HOURLY = "funding_rate_bps_hourly"
F_HOURS_TO_FUNDING = "hours_to_funding"
F_FUNDING_CARRY_8H_CC = "funding_carry_8h_cc"
F_BASIS_CC = "basis_cc"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Size the effective-spread feature is priced at. Ten contracts is a real order
#: on Kalshi rather than a notional one-lot, so the number reflects what this
#: platform would actually pay rather than the best case at the touch.
REFERENCE_SIZE_QC: Final[int] = 10 * QC_PER_CONTRACT

_MS_PER_SECOND: Final[int] = 1_000
_MS_PER_MINUTE: Final[int] = 60_000
_MS_PER_HOUR: Final[int] = 3_600_000

_W_30S: Final[int] = 30 * _MS_PER_SECOND
_W_60S: Final[int] = 60 * _MS_PER_SECOND
_W_1M: Final[int] = _MS_PER_MINUTE
_W_5M: Final[int] = 5 * _MS_PER_MINUTE
_W_15M: Final[int] = 15 * _MS_PER_MINUTE
_W_60M: Final[int] = 60 * _MS_PER_MINUTE
_W_120M: Final[int] = 120 * _MS_PER_MINUTE

#: Minimum observations before a dispersion estimate is reported rather than
#: imputed. Two points always have a standard deviation and it always means
#: nothing.
_MIN_DISPERSION_POINTS: Final[int] = 4

#: Funding is quoted per hour; eight hours is the conventional carry horizon and
#: the one the perp strategies size against.
_CARRY_HOURS: Final[int] = 8


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar.

    ``ts_ms`` is the bar's **close** time, not its open. The distinction is the
    whole reason :data:`kat.features.spec.LAG_CANDLE_MS` exists: a bar covering
    12:00:00-12:00:59 is an observation about 12:00:59 and is in our hands some
    seconds after that. Storing the open time here and lagging from it would
    understate the delay by a whole period.
    """

    ts_ms: int
    close_cc: int
    open_cc: int | None = None
    high_cc: int | None = None
    low_cc: int | None = None
    volume_qc: int = 0
    open_interest_qc: int | None = None


class BarSeries:
    """Time-ordered bars with point-in-time access.

    Every read is filtered by ``as_of_ms`` and the caller's availability lag, so
    a bar that had not yet been published cannot enter a calculation even if it
    is sitting in this object. Lookups are binary searches over a cached
    timestamp list rather than scans, because the training pipeline calls them
    once per feature per decision time and there are a lot of decision times.
    """

    __slots__ = ("_bars", "_ts", "period_ms")

    def __init__(self, bars: Iterable[Bar], *, period_ms: int) -> None:
        if period_ms <= 0:
            raise DataQualityError(f"bar period must be positive, got {period_ms}")
        ordered = sorted(bars, key=lambda b: b.ts_ms)
        for prev, nxt in zip(ordered, ordered[1:], strict=False):
            if prev.ts_ms == nxt.ts_ms:
                raise DataQualityError(f"duplicate bar at ts={prev.ts_ms}")
        self.period_ms = period_ms
        self._bars: list[Bar] = ordered
        self._ts: list[int] = [b.ts_ms for b in ordered]

    def __len__(self) -> int:
        return len(self._bars)

    def available_count(self, *, as_of_ms: int, lag_ms: int) -> int:
        """Number of leading bars publishable by ``as_of_ms``."""
        return bisect.bisect_right(self._ts, as_of_ms - lag_ms)

    def latest(self, *, as_of_ms: int, lag_ms: int) -> Bar | None:
        hi = self.available_count(as_of_ms=as_of_ms, lag_ms=lag_ms)
        return self._bars[hi - 1] if hi else None

    def window(self, *, as_of_ms: int, lag_ms: int, span_ms: int) -> list[Bar]:
        """Available bars whose close falls in the trailing ``span_ms``."""
        hi = self.available_count(as_of_ms=as_of_ms, lag_ms=lag_ms)
        if hi == 0:
            return []
        lo = bisect.bisect_left(self._ts, self._ts[hi - 1] - span_ms, 0, hi)
        return self._bars[lo:hi]

    def at_or_before(self, ts_ms: int, *, as_of_ms: int, lag_ms: int) -> Bar | None:
        hi = self.available_count(as_of_ms=as_of_ms, lag_ms=lag_ms)
        idx = bisect.bisect_right(self._ts, ts_ms, 0, hi)
        return self._bars[idx - 1] if idx else None


@dataclass(frozen=True, slots=True)
class TradePrint:
    """One executed trade.

    ``taker_is_buy`` is ``None`` when the exchange did not tell us which side
    lifted. It is left unknown rather than guessed at ingest, because the guess
    belongs where it can be documented -- see :func:`_f_trade_imbalance_5m`.
    """

    ts_ms: int
    price_cc: int
    qty_qc: int
    taker_is_buy: bool | None = None


class TradeSeries:
    """Time-ordered trade prints with point-in-time access."""

    __slots__ = ("_prints", "_ts")

    def __init__(self, prints: Iterable[TradePrint]) -> None:
        ordered = sorted(prints, key=lambda t: t.ts_ms)
        self._prints: list[TradePrint] = ordered
        self._ts: list[int] = [t.ts_ms for t in ordered]

    def __len__(self) -> int:
        return len(self._prints)

    def window(self, *, as_of_ms: int, lag_ms: int, span_ms: int) -> list[TradePrint]:
        cutoff = as_of_ms - lag_ms
        hi = bisect.bisect_right(self._ts, cutoff)
        lo = bisect.bisect_left(self._ts, cutoff - span_ms, 0, hi)
        return self._prints[lo:hi]


class MidSeries:
    """Book midpoints stamped with the moment the book message was applied.

    This is the only high-frequency price history that carries a zero
    availability lag, which makes it the only honest source for sub-minute
    momentum and volatility. The live engine appends to it on every book update;
    the training pipeline reconstructs it from stored book snapshots. Both go
    through the same object, so the sub-minute features cannot mean one thing in
    research and another in production.
    """

    __slots__ = ("_mids", "_ts")

    def __init__(self, points: Iterable[tuple[int, int]] = ()) -> None:
        ordered = sorted(points, key=lambda p: p[0])
        self._ts: list[int] = [p[0] for p in ordered]
        self._mids: list[int] = [p[1] for p in ordered]

    def __len__(self) -> int:
        return len(self._ts)

    def append(self, ts_ms: int, mid_cc: int) -> None:
        if self._ts and ts_ms < self._ts[-1]:
            raise DataQualityError(f"out-of-order mid at ts={ts_ms} (last={self._ts[-1]})")
        self._ts.append(ts_ms)
        self._mids.append(mid_cc)

    def trim_before(self, ts_ms: int) -> None:
        """Drop points older than ``ts_ms``. Bounds memory in a long live run."""
        idx = bisect.bisect_left(self._ts, ts_ms)
        if idx:
            del self._ts[:idx]
            del self._mids[:idx]

    def window(self, *, as_of_ms: int, lag_ms: int, span_ms: int) -> list[tuple[int, int]]:
        cutoff = as_of_ms - lag_ms
        hi = bisect.bisect_right(self._ts, cutoff)
        lo = bisect.bisect_left(self._ts, cutoff - span_ms, 0, hi)
        return list(zip(self._ts[lo:hi], self._mids[lo:hi], strict=True))

    def at_or_before(self, ts_ms: int, *, as_of_ms: int, lag_ms: int) -> tuple[int, int] | None:
        hi = bisect.bisect_right(self._ts, as_of_ms - lag_ms)
        idx = bisect.bisect_right(self._ts, ts_ms, 0, hi)
        if idx == 0:
            return None
        return self._ts[idx - 1], self._mids[idx - 1]


@dataclass(slots=True)
class FeatureInputs:
    """Everything one market's features are computed from, at one decision time.

    Assembled identically by the training pipeline and by the live engine. The
    per-source timestamps are separate fields rather than one "updated" time
    because they are what the availability arithmetic is made of: a stale ticker
    alongside a fresh book must not make book features look stale, nor the other
    way round.
    """

    ticker: str
    market_type: MarketType

    book: OrderBook | None = None
    book_ts_ms: int = 0
    mids: MidSeries | None = None

    bars: BarSeries | None = None
    trades: TradeSeries | None = None

    ticker_ts_ms: int = 0
    volume_qc: int | None = None
    open_interest_qc: int | None = None
    last_trade_cc: int | None = None
    mark_price_cc: int | None = None

    funding_ts_ms: int = 0
    funding_rate_hourly: float | None = None
    next_funding_ms: int | None = None

    #: When the market stops trading. ``None`` for perpetuals, and for event
    #: contracts whose definition we have not loaded -- in which case the term
    #: features are absent and, being required, abort the vector.
    close_ts_ms: int | None = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: MarketSnapshot,
        *,
        bars: BarSeries | None = None,
        trades: TradeSeries | None = None,
        mids: MidSeries | None = None,
        close_ts_ms: int | None = None,
    ) -> FeatureInputs:
        """Adapt live market state into computation inputs.

        ``close_ts_ms`` falls back to the snapshot metadata written by the
        market-definition loader, which is where the event-contract close time
        lives once a market has been subscribed.
        """
        close = close_ts_ms
        if close is None:
            meta_close = snapshot.metadata.get("close_ts_ms")
            close = meta_close if isinstance(meta_close, int) else None
        return cls(
            ticker=snapshot.ticker,
            market_type=snapshot.market_type,
            book=snapshot.book,
            book_ts_ms=snapshot.book_updated_ms or snapshot.updated_ms,
            mids=mids,
            bars=bars,
            trades=trades,
            ticker_ts_ms=snapshot.updated_ms,
            volume_qc=snapshot.volume_qc,
            open_interest_qc=snapshot.open_interest_qc,
            last_trade_cc=snapshot.last_trade_cc,
            mark_price_cc=snapshot.mark_price_cc,
            funding_ts_ms=snapshot.updated_ms,
            funding_rate_hourly=snapshot.funding_rate,
            next_funding_ms=snapshot.next_funding_ms,
            close_ts_ms=close,
        )


# ---------------------------------------------------------------------------
# Calculation scratch space
# ---------------------------------------------------------------------------


class _Calc:
    """Derived quantities shared between features, computed at most once.

    Several features need the same book midpoint or the same 60-minute bar
    window. Recomputing them per feature would be a constant factor on the
    inference hot path for no benefit, so they are memoised for the duration of
    one :meth:`FeatureComputer.compute` call and thrown away with it -- there is
    no cross-call cache, because a cache keyed on anything other than the exact
    input set is how stale data reaches a live decision.
    """

    __slots__ = ("_cache", "as_of_ms", "inputs")

    def __init__(self, inputs: FeatureInputs, as_of_ms: int) -> None:
        self.inputs = inputs
        self.as_of_ms = as_of_ms
        self._cache: dict[str, Any] = {}

    # -- book ---------------------------------------------------------------

    @property
    def book(self) -> OrderBook | None:
        book = self.inputs.book
        if book is None:
            return None
        # A crossed book is a broken book: the feed is mid-update, or two
        # snapshots have been interleaved. Every microstructure number derived
        # from it would be nonsense with a plausible magnitude, which is worse
        # than an absent feature.
        return None if book.is_crossed else book

    @property
    def book_ts_ms(self) -> int:
        return self.inputs.book_ts_ms

    @property
    def two_sided(self) -> bool:
        book = self.book
        return book is not None and book.best_bid_cc is not None and book.best_ask_cc is not None

    @property
    def mid_cc(self) -> int | None:
        book = self.book
        return book.mid_cc if book is not None else None

    # -- bars ---------------------------------------------------------------

    @property
    def bars(self) -> BarSeries | None:
        return self.inputs.bars

    @property
    def latest_bar(self) -> Bar | None:
        key = "latest_bar"
        if key not in self._cache:
            series = self.bars
            self._cache[key] = (
                None
                if series is None
                else series.latest(as_of_ms=self.as_of_ms, lag_ms=LAG_CANDLE_MS)
            )
        return self._cache[key]  # type: ignore[no-any-return]

    def bar_window(self, span_ms: int) -> list[Bar]:
        key = f"bars:{span_ms}"
        if key not in self._cache:
            series = self.bars
            self._cache[key] = (
                []
                if series is None
                else series.window(as_of_ms=self.as_of_ms, lag_ms=LAG_CANDLE_MS, span_ms=span_ms)
            )
        return self._cache[key]  # type: ignore[no-any-return]

    def bar_at_or_before(self, ts_ms: int) -> Bar | None:
        series = self.bars
        if series is None:
            return None
        return series.at_or_before(ts_ms, as_of_ms=self.as_of_ms, lag_ms=LAG_CANDLE_MS)

    def consecutive_close_changes(self, span_ms: int) -> list[int]:
        """Close-to-close changes between *adjacent* bars only.

        A gap in the series -- a halted market, a dropped poll -- makes the
        difference across it a multi-period move. Including it would inflate a
        one-minute volatility estimate by whatever the gap happened to be, so
        non-adjacent pairs are dropped rather than silently averaged in.
        """
        key = f"dcloses:{span_ms}"
        if key not in self._cache:
            series = self.bars
            window = self.bar_window(span_ms)
            period = series.period_ms if series is not None else 0
            changes: list[int] = []
            for prev, nxt in zip(window, window[1:], strict=False):
                if nxt.ts_ms - prev.ts_ms == period:
                    changes.append(nxt.close_cc - prev.close_cc)
            self._cache[key] = changes
        return self._cache[key]  # type: ignore[no-any-return]

    # -- mids ---------------------------------------------------------------

    def mid_window(self, span_ms: int) -> list[tuple[int, int]]:
        key = f"mids:{span_ms}"
        if key not in self._cache:
            series = self.inputs.mids
            self._cache[key] = (
                []
                if series is None
                else series.window(as_of_ms=self.as_of_ms, lag_ms=LAG_BOOK_MS, span_ms=span_ms)
            )
        return self._cache[key]  # type: ignore[no-any-return]

    # -- trades -------------------------------------------------------------

    def trade_window(self, span_ms: int) -> list[TradePrint]:
        key = f"trades:{span_ms}"
        if key not in self._cache:
            series = self.inputs.trades
            self._cache[key] = (
                []
                if series is None
                else series.window(as_of_ms=self.as_of_ms, lag_ms=LAG_TICKER_MS, span_ms=span_ms)
            )
        return self._cache[key]  # type: ignore[no-any-return]


#: A feature function returns ``(value, info_ts_ms)`` or ``None`` when the value
#: cannot be computed from the available data. Returning ``None`` is a normal
#: outcome, not an error: the spec decides whether absence is fatal.
_Feature = Callable[[_Calc], "tuple[float, int] | None"]


# ---------------------------------------------------------------------------
# Microstructure
# ---------------------------------------------------------------------------


def _f_spread_cc(c: _Calc) -> tuple[float, int] | None:
    book = c.book
    if book is None:
        return None
    spread = book.spread_cc
    return None if spread is None else (float(spread), c.book_ts_ms)


def _f_relative_spread(c: _Calc) -> tuple[float, int] | None:
    book = c.book
    if book is None:
        return None
    spread = book.spread_cc
    mid = book.mid_cc
    if spread is None or not mid:
        return None
    return spread / mid, c.book_ts_ms


def _f_microprice_offset_cc(c: _Calc) -> tuple[float, int] | None:
    """Size-weighted touch price minus the midpoint.

    The midpoint assumes both sides are equally likely to be hit. When 900
    contracts rest on the bid against 20 on the ask, the next trade is far more
    likely to be a buy, and the microprice says so.
    """
    book = c.book
    if not c.two_sided:
        return None
    assert book is not None
    bid, ask = book.best_bid_cc, book.best_ask_cc
    assert bid is not None and ask is not None
    bid_sz, ask_sz = book.best_bid_size_qc, book.best_ask_size_qc
    total = bid_sz + ask_sz
    if total <= 0:
        return None
    micro = (bid * ask_sz + ask * bid_sz) / total
    return micro - book.mid_cc, c.book_ts_ms  # type: ignore[operator]


def _f_book_imbalance(c: _Calc) -> tuple[float, int] | None:
    book = c.book
    if book is None:
        return None
    bid = book.depth_qc(is_bid=True, levels=5)
    ask = book.depth_qc(is_bid=False, levels=5)
    total = bid + ask
    if total <= 0:
        return None
    return (bid - ask) / total, c.book_ts_ms


def _f_book_pressure_l1(c: _Calc) -> tuple[float, int] | None:
    book = c.book
    if not c.two_sided:
        return None
    assert book is not None
    bid, ask = book.best_bid_size_qc, book.best_ask_size_qc
    total = bid + ask
    if total <= 0:
        return None
    return (bid - ask) / total, c.book_ts_ms


def _f_effective_spread_cc(c: _Calc) -> tuple[float, int] | None:
    """Half the round-trip cost of :data:`REFERENCE_SIZE_QC`, in centicents.

    Priced against the depth that is actually there: if the book cannot absorb
    the reference size, the cost is measured over everything available rather
    than reported as missing, because a thin book is expensive, not unknown.
    """
    book = c.book
    if not c.two_sided:
        return None
    assert book is not None
    buy_qty = min(REFERENCE_SIZE_QC, book.total_depth_qc(is_bid=False))
    sell_qty = min(REFERENCE_SIZE_QC, book.total_depth_qc(is_bid=True))
    if buy_qty <= 0 or sell_qty <= 0:
        return None
    buy_vwap = book.vwap_cc(buy_qty, is_buy=True)
    sell_vwap = book.vwap_cc(sell_qty, is_buy=False)
    if buy_vwap is None or sell_vwap is None:
        return None
    return (buy_vwap - sell_vwap) / 2.0, c.book_ts_ms


def _f_depth_total_log(c: _Calc) -> tuple[float, int] | None:
    """``log1p`` of total two-sided depth in contracts.

    Logged because depth spans four orders of magnitude across Kalshi markets
    and a linear model fed the raw number learns almost nothing except which
    markets are large.
    """
    book = c.book
    if book is None:
        return None
    total = book.total_depth_qc(is_bid=True) + book.total_depth_qc(is_bid=False)
    return math.log1p(total / QC_PER_CONTRACT), c.book_ts_ms


def _f_depth_concentration(c: _Calc) -> tuple[float, int] | None:
    """Share of five-level depth resting at the touch.

    High concentration means the book looks deep but evaporates one level down:
    the resilience the size suggests is not there.
    """
    book = c.book
    if book is None:
        return None
    top = book.depth_qc(is_bid=True, levels=1) + book.depth_qc(is_bid=False, levels=1)
    five = book.depth_qc(is_bid=True, levels=5) + book.depth_qc(is_bid=False, levels=5)
    if five <= 0:
        return None
    return top / five, c.book_ts_ms


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def _mid_change(c: _Calc, span_ms: int) -> tuple[float, int] | None:
    series = c.inputs.mids
    if series is None:
        return None
    window = c.mid_window(span_ms)
    if len(window) < 2:
        return None
    latest_ts, latest_mid = window[-1]
    prior = series.at_or_before(latest_ts - span_ms, as_of_ms=c.as_of_ms, lag_ms=LAG_BOOK_MS)
    if prior is None or prior[0] == latest_ts:
        return None
    return float(latest_mid - prior[1]), latest_ts


def _f_mid_change_30s_cc(c: _Calc) -> tuple[float, int] | None:
    return _mid_change(c, _W_30S)


def _bar_change(c: _Calc, span_ms: int) -> tuple[float, int] | None:
    latest = c.latest_bar
    if latest is None:
        return None
    prior = c.bar_at_or_before(latest.ts_ms - span_ms)
    if prior is None or prior.ts_ms == latest.ts_ms:
        return None
    return float(latest.close_cc - prior.close_cc), latest.ts_ms


def _f_mid_change_1m_cc(c: _Calc) -> tuple[float, int] | None:
    return _bar_change(c, _W_1M)


def _f_mid_change_5m_cc(c: _Calc) -> tuple[float, int] | None:
    return _bar_change(c, _W_5M)


def _f_mid_change_15m_cc(c: _Calc) -> tuple[float, int] | None:
    return _bar_change(c, _W_15M)


def _f_mid_change_60m_cc(c: _Calc) -> tuple[float, int] | None:
    return _bar_change(c, _W_60M)


def _mean_reversion(c: _Calc, span_ms: int) -> tuple[float, int] | None:
    """Latest close minus the mean close over the window.

    Positive means the price is stretched above its own recent average. Whether
    that is momentum or something to fade is the model's problem; the feature
    only states the displacement.
    """
    window = c.bar_window(span_ms)
    if len(window) < _MIN_DISPERSION_POINTS:
        return None
    latest = window[-1]
    mean = statistics.fmean(b.close_cc for b in window)
    return latest.close_cc - mean, latest.ts_ms


def _f_mean_reversion_15m_cc(c: _Calc) -> tuple[float, int] | None:
    return _mean_reversion(c, _W_15M)


def _f_mean_reversion_60m_cc(c: _Calc) -> tuple[float, int] | None:
    return _mean_reversion(c, _W_60M)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def _f_mid_vol_60s_cc(c: _Calc) -> tuple[float, int] | None:
    """Standard deviation of book midpoints over the last minute.

    Dispersion of the level rather than of returns, because at sub-minute
    horizons the mid moves in one-centicent steps and a return series is mostly
    zeros with occasional ticks.
    """
    window = c.mid_window(_W_60S)
    if len(window) < _MIN_DISPERSION_POINTS:
        return None
    return statistics.stdev(m for _, m in window), window[-1][0]


def _realized_vol(c: _Calc, span_ms: int) -> tuple[float, int] | None:
    changes = c.consecutive_close_changes(span_ms)
    if len(changes) < _MIN_DISPERSION_POINTS:
        return None
    window = c.bar_window(span_ms)
    return statistics.stdev(changes), window[-1].ts_ms


def _f_realized_vol_5m_cc(c: _Calc) -> tuple[float, int] | None:
    return _realized_vol(c, _W_15M)


def _f_realized_vol_60m_cc(c: _Calc) -> tuple[float, int] | None:
    return _realized_vol(c, _W_60M)


def _f_vol_ratio_15m_60m(c: _Calc) -> tuple[float, int] | None:
    """Short-horizon volatility relative to long-horizon.

    Above one is a market that has just started moving. This is the cheapest
    regime signal there is, and unlike either level on its own it is comparable
    across markets of wildly different activity.
    """
    short = _realized_vol(c, _W_15M)
    long = _realized_vol(c, _W_60M)
    if short is None or long is None or long[0] <= 0.0:
        return None
    return short[0] / long[0], max(short[1], long[1])


def _f_range_vol_15m_cc(c: _Calc) -> tuple[float, int] | None:
    """Parkinson-style range estimator, in additive centicents.

    The textbook estimator uses ``log(high/low)``, which is unusable here: a
    contract ranging $0.01-$0.02 would register more volatility than one ranging
    $0.40-$0.60. The additive form is the correct analogue in probability space.
    """
    window = [b for b in c.bar_window(_W_15M) if b.high_cc is not None and b.low_cc is not None]
    if len(window) < _MIN_DISPERSION_POINTS:
        return None
    mean_sq = statistics.fmean((b.high_cc - b.low_cc) ** 2 for b in window)  # type: ignore[operator]
    return math.sqrt(mean_sq / (4.0 * math.log(2.0))), window[-1].ts_ms


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


def _f_volume_5m_qc(c: _Calc) -> tuple[float, int] | None:
    window = c.bar_window(_W_5M)
    if not window:
        return None
    return float(sum(b.volume_qc for b in window)), window[-1].ts_ms


def _f_volume_zscore_5m(c: _Calc) -> tuple[float, int] | None:
    """Trailing five-minute volume against this market's own recent distribution.

    Standardised per market rather than across markets: 500 contracts is a storm
    in one market and a rounding error in another, and the interesting event is
    a market becoming unusually active *for itself*.

    The rolling sums are taken over bar positions rather than wall-clock offsets.
    Where the series has gaps a window therefore spans more than five minutes,
    which biases the reference distribution upward and the resulting z-score
    downward -- the conservative direction for a feature that gates trading.
    """
    window = c.bar_window(_W_120M)
    series = c.bars
    if series is None or not window:
        return None
    k = max(1, _W_5M // series.period_ms)
    if len(window) < k + _MIN_DISPERSION_POINTS:
        return None
    vols = [b.volume_qc for b in window]
    sums = [sum(vols[i : i + k]) for i in range(len(vols) - k + 1)]
    spread = statistics.stdev(sums)
    if spread <= 0.0:
        return None
    return (sums[-1] - statistics.fmean(sums)) / spread, window[-1].ts_ms


def _f_trade_imbalance_5m(c: _Calc) -> tuple[float, int] | None:
    """Signed taker pressure in [-1, 1]; +1 is every contract lifting the offer.

    Where the exchange reports the aggressor, it is used. Where it does not, the
    tick rule signs a print by its move against the previous one, and a print at
    an unchanged price stays unsigned: it still counts in the denominator, so
    unknowable flow dilutes the reading toward zero instead of being invented in
    one direction.
    """
    window = c.trade_window(_W_5M)
    if not window:
        return None
    signed = 0
    total = 0
    prev_price: int | None = None
    for print_ in window:
        total += print_.qty_qc
        if print_.taker_is_buy is not None:
            signed += print_.qty_qc if print_.taker_is_buy else -print_.qty_qc
        elif prev_price is not None and print_.price_cc != prev_price:
            signed += print_.qty_qc if print_.price_cc > prev_price else -print_.qty_qc
        prev_price = print_.price_cc
    if total <= 0:
        return None
    return signed / total, window[-1].ts_ms


def _oi_pair(c: _Calc) -> tuple[Bar, Bar] | None:
    latest = c.latest_bar
    if latest is None or latest.open_interest_qc is None:
        return None
    prior = c.bar_at_or_before(latest.ts_ms - _W_5M)
    if prior is None or prior.open_interest_qc is None or prior.ts_ms == latest.ts_ms:
        return None
    return latest, prior


def _f_oi_change_5m_qc(c: _Calc) -> tuple[float, int] | None:
    """Change in open interest.

    Distinguishes new risk being put on from existing risk changing hands, which
    volume alone cannot: a large print against flat open interest is a transfer,
    the same print against rising open interest is fresh positioning.
    """
    pair = _oi_pair(c)
    if pair is None:
        return None
    latest, prior = pair
    assert latest.open_interest_qc is not None and prior.open_interest_qc is not None
    return float(latest.open_interest_qc - prior.open_interest_qc), latest.ts_ms


def _f_oi_change_rate_5m(c: _Calc) -> tuple[float, int] | None:
    pair = _oi_pair(c)
    if pair is None:
        return None
    latest, prior = pair
    assert latest.open_interest_qc is not None and prior.open_interest_qc is not None
    if prior.open_interest_qc <= 0:
        return None
    return (latest.open_interest_qc - prior.open_interest_qc) / prior.open_interest_qc, latest.ts_ms


# ---------------------------------------------------------------------------
# Level
# ---------------------------------------------------------------------------


def _f_price_level(c: _Calc) -> tuple[float, int] | None:
    """Midpoint as an implied probability in [0, 1]. Event contracts only."""
    mid = c.mid_cc
    if mid is None:
        return None
    return mid / CC_PER_DOLLAR, c.book_ts_ms


def _f_distance_from_half(c: _Calc) -> tuple[float, int] | None:
    """How far from a coin flip the market is.

    Not redundant with the level: the payoff asymmetry, the tail behaviour and
    the cost of being wrong are all symmetric about $0.50, and forcing a model
    to learn that symmetry from the level alone wastes the sample.
    """
    mid = c.mid_cc
    if mid is None:
        return None
    return abs(mid / CC_PER_DOLLAR - 0.5), c.book_ts_ms


def _range_60m(c: _Calc) -> tuple[Bar, int, int] | None:
    window = c.bar_window(_W_60M)
    if len(window) < _MIN_DISPERSION_POINTS:
        return None
    highs = [b.high_cc if b.high_cc is not None else b.close_cc for b in window]
    lows = [b.low_cc if b.low_cc is not None else b.close_cc for b in window]
    return window[-1], max(highs), min(lows)


def _f_range_position_60m(c: _Calc) -> tuple[float, int] | None:
    """Where the last close sits inside the hour's range, 0 at the low, 1 at the high."""
    triple = _range_60m(c)
    if triple is None:
        return None
    latest, high, low = triple
    width = high - low
    if width <= 0:
        return None
    return (latest.close_cc - low) / width, latest.ts_ms


def _f_range_width_60m_cc(c: _Calc) -> tuple[float, int] | None:
    triple = _range_60m(c)
    if triple is None:
        return None
    latest, high, low = triple
    return float(high - low), latest.ts_ms


# ---------------------------------------------------------------------------
# Term (event contracts only)
# ---------------------------------------------------------------------------


def _time_to_expiry_ms(c: _Calc) -> int | None:
    close = c.inputs.close_ts_ms
    if close is None:
        return None
    return max(0, close - c.as_of_ms)


def _f_time_to_expiry_ms(c: _Calc) -> tuple[float, int] | None:
    remaining = _time_to_expiry_ms(c)
    if remaining is None:
        return None
    # Pure clock feature: the information timestamp is the decision time itself,
    # because nothing was observed to produce it.
    return float(remaining), c.as_of_ms


def _f_log_time_to_expiry_h(c: _Calc) -> tuple[float, int] | None:
    """``log1p`` of hours remaining.

    Convergence to settlement is not linear in time. The difference between one
    hour and two hours left is enormous; the difference between thirty days and
    thirty-one is nothing, and a linear clock feature forces a model to spend
    capacity discovering that.
    """
    remaining = _time_to_expiry_ms(c)
    if remaining is None:
        return None
    return math.log1p(remaining / _MS_PER_HOUR), c.as_of_ms


def _f_expiry_scaled_move_cc(c: _Calc) -> tuple[float, int] | None:
    """Volatility scaled by the square root of the time remaining.

    How far the price can plausibly travel before settlement, in centicents.
    This is what decides whether a contract five points from the money is a
    near-certainty or a coin flip, and neither volatility nor time can say it
    alone. The clock contributes no information timestamp, so the value carries
    the candle timestamp and the candle lag.
    """
    remaining = _time_to_expiry_ms(c)
    vol = _realized_vol(c, _W_15M)
    if remaining is None or vol is None:
        return None
    minutes = remaining / _MS_PER_MINUTE
    return vol[0] * math.sqrt(minutes), vol[1]


# ---------------------------------------------------------------------------
# Carry (perpetuals only)
# ---------------------------------------------------------------------------


def _f_funding_rate_bps_hourly(c: _Calc) -> tuple[float, int] | None:
    """Hourly funding rate in basis points.

    Signed from the perspective of a long: positive means longs pay. It is a
    real cash flow against a held position and belongs in the model rather than
    only in the cost calculation, because it is also information about where
    positioning is crowded.
    """
    rate = c.inputs.funding_rate_hourly
    if rate is None or not math.isfinite(rate):
        return None
    return rate * 10_000.0, c.inputs.funding_ts_ms


def _f_hours_to_funding(c: _Calc) -> tuple[float, int] | None:
    nxt = c.inputs.next_funding_ms
    if nxt is None:
        return None
    return max(0.0, (nxt - c.as_of_ms) / _MS_PER_HOUR), c.inputs.funding_ts_ms


def _f_funding_carry_8h_cc(c: _Calc) -> tuple[float, int] | None:
    """Expected funding paid over eight hours, expressed in centicents of price.

    Converting the rate into the same unit as the price is what makes it
    comparable with an expected move: a carry of 30cc against an expected move
    of 10cc is a trade that loses on average however right the direction is.
    """
    rate = c.inputs.funding_rate_hourly
    mid = c.mid_cc
    if rate is None or mid is None or not math.isfinite(rate):
        return None
    return rate * _CARRY_HOURS * mid, c.inputs.funding_ts_ms


def _f_basis_cc(c: _Calc) -> tuple[float, int] | None:
    """Exchange mark minus book midpoint.

    The mark is what positions are valued and liquidated at; the mid is where
    they can be traded. A persistent gap between them is either an opportunity
    or a warning that the book has gone thin, and both are worth knowing before
    sizing.
    """
    mark = c.inputs.mark_price_cc
    mid = c.mid_cc
    if mark is None or mid is None:
        return None
    return float(mark - mid), max(c.inputs.ticker_ts_ms, c.book_ts_ms)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

_BOTH: Final[tuple[MarketType, ...]] = (MarketType.EVENT, MarketType.PERP)
_EVENT_ONLY: Final[tuple[MarketType, ...]] = (MarketType.EVENT,)
_PERP_ONLY: Final[tuple[MarketType, ...]] = (MarketType.PERP,)

#: Every feature this module can compute, paired with the function that computes
#: it. The pairing lives in one structure so a spec cannot be declared without an
#: implementation, nor an implementation shipped without a declared lag.
_CATALOGUE: Final[tuple[tuple[FeatureSpec, _Feature], ...]] = (
    # -- microstructure ------------------------------------------------------
    (
        FeatureSpec(
            name=F_SPREAD_CC,
            description="Best ask minus best bid, in centicents.",
            family=FeatureFamily.MICROSTRUCTURE,
            source=FeatureSource.BOOK,
            market_types=_BOTH,
            clip_low=0.0,
            clip_high=float(CC_PER_DOLLAR),
        ),
        _f_spread_cc,
    ),
    (
        FeatureSpec(
            name=F_RELATIVE_SPREAD,
            description="Spread as a fraction of the midpoint.",
            family=FeatureFamily.MICROSTRUCTURE,
            source=FeatureSource.BOOK,
            market_types=_BOTH,
            clip_low=0.0,
            clip_high=1.0,
        ),
        _f_relative_spread,
    ),
    (
        FeatureSpec(
            name=F_MICROPRICE_OFFSET_CC,
            description="Size-weighted touch price minus the midpoint, in centicents.",
            family=FeatureFamily.MICROSTRUCTURE,
            source=FeatureSource.BOOK,
            market_types=_BOTH,
            clip_low=-float(CC_PER_DOLLAR),
            clip_high=float(CC_PER_DOLLAR),
        ),
        _f_microprice_offset_cc,
    ),
    (
        FeatureSpec(
            name=F_BOOK_IMBALANCE,
            description="(bid depth - ask depth) / total depth over the top five levels.",
            family=FeatureFamily.MICROSTRUCTURE,
            source=FeatureSource.BOOK,
            market_types=_BOTH,
            clip_low=-1.0,
            clip_high=1.0,
        ),
        _f_book_imbalance,
    ),
    (
        FeatureSpec(
            name=F_BOOK_PRESSURE_L1,
            description="Same imbalance restricted to the touch.",
            family=FeatureFamily.MICROSTRUCTURE,
            source=FeatureSource.BOOK,
            market_types=_BOTH,
            clip_low=-1.0,
            clip_high=1.0,
        ),
        _f_book_pressure_l1,
    ),
    (
        FeatureSpec(
            name=F_EFFECTIVE_SPREAD_CC,
            description=(
                "Half the round-trip cost of trading the reference size, from walking the book. "
                "Missing values impute to the pessimistic extreme, never to zero."
            ),
            family=FeatureFamily.MICROSTRUCTURE,
            source=FeatureSource.BOOK,
            market_types=_BOTH,
            fill_value=float(CC_PER_DOLLAR),
            clip_low=0.0,
            clip_high=float(CC_PER_DOLLAR),
        ),
        _f_effective_spread_cc,
    ),
    (
        FeatureSpec(
            name=F_DEPTH_TOTAL_LOG,
            description="log1p of total two-sided resting depth, in contracts.",
            family=FeatureFamily.MICROSTRUCTURE,
            source=FeatureSource.BOOK,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=0.0,
        ),
        _f_depth_total_log,
    ),
    (
        FeatureSpec(
            name=F_DEPTH_CONCENTRATION,
            description="Fraction of five-level depth resting at the touch.",
            family=FeatureFamily.MICROSTRUCTURE,
            source=FeatureSource.BOOK,
            market_types=_BOTH,
            fill_value=1.0,
            clip_low=0.0,
            clip_high=1.0,
        ),
        _f_depth_concentration,
    ),
    # -- momentum ------------------------------------------------------------
    (
        FeatureSpec(
            name=F_MID_CHANGE_30S_CC,
            description="Midpoint change over 30 seconds, in centicents.",
            family=FeatureFamily.MOMENTUM,
            source=FeatureSource.BOOK,
            window_ms=_W_30S,
            market_types=_BOTH,
            fill_value=0.0,
        ),
        _f_mid_change_30s_cc,
    ),
    (
        FeatureSpec(
            name=F_MID_CHANGE_1M_CC,
            description="Close-to-close change over one minute, in centicents.",
            family=FeatureFamily.MOMENTUM,
            source=FeatureSource.CANDLES,
            window_ms=_W_1M,
            market_types=_BOTH,
            fill_value=0.0,
        ),
        _f_mid_change_1m_cc,
    ),
    (
        FeatureSpec(
            name=F_MID_CHANGE_5M_CC,
            description="Close-to-close change over five minutes, in centicents.",
            family=FeatureFamily.MOMENTUM,
            source=FeatureSource.CANDLES,
            window_ms=_W_5M,
            market_types=_BOTH,
            fill_value=0.0,
        ),
        _f_mid_change_5m_cc,
    ),
    (
        FeatureSpec(
            name=F_MID_CHANGE_15M_CC,
            description="Close-to-close change over fifteen minutes, in centicents.",
            family=FeatureFamily.MOMENTUM,
            source=FeatureSource.CANDLES,
            window_ms=_W_15M,
            market_types=_BOTH,
            fill_value=0.0,
        ),
        _f_mid_change_15m_cc,
    ),
    (
        FeatureSpec(
            name=F_MID_CHANGE_60M_CC,
            description="Close-to-close change over sixty minutes, in centicents.",
            family=FeatureFamily.MOMENTUM,
            source=FeatureSource.CANDLES,
            window_ms=_W_60M,
            market_types=_BOTH,
            fill_value=0.0,
        ),
        _f_mid_change_60m_cc,
    ),
    (
        FeatureSpec(
            name=F_MEAN_REVERSION_15M_CC,
            description="Last close minus the mean close over fifteen minutes.",
            family=FeatureFamily.MOMENTUM,
            source=FeatureSource.CANDLES,
            window_ms=_W_15M,
            market_types=_BOTH,
            fill_value=0.0,
        ),
        _f_mean_reversion_15m_cc,
    ),
    (
        FeatureSpec(
            name=F_MEAN_REVERSION_60M_CC,
            description="Last close minus the mean close over sixty minutes.",
            family=FeatureFamily.MOMENTUM,
            source=FeatureSource.CANDLES,
            window_ms=_W_60M,
            market_types=_BOTH,
            fill_value=0.0,
        ),
        _f_mean_reversion_60m_cc,
    ),
    # -- volatility ----------------------------------------------------------
    (
        FeatureSpec(
            name=F_MID_VOL_60S_CC,
            description="Standard deviation of book midpoints over the last minute.",
            family=FeatureFamily.VOLATILITY,
            source=FeatureSource.BOOK,
            window_ms=_W_60S,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=0.0,
        ),
        _f_mid_vol_60s_cc,
    ),
    (
        FeatureSpec(
            name=F_REALIZED_VOL_5M_CC,
            description=(
                "Standard deviation of one-minute close-to-close changes over the last "
                "fifteen minutes, in centicents. Non-adjacent bars are excluded."
            ),
            family=FeatureFamily.VOLATILITY,
            source=FeatureSource.CANDLES,
            window_ms=_W_15M,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=0.0,
        ),
        _f_realized_vol_5m_cc,
    ),
    (
        FeatureSpec(
            name=F_REALIZED_VOL_60M_CC,
            description="Same estimator over the last sixty minutes.",
            family=FeatureFamily.VOLATILITY,
            source=FeatureSource.CANDLES,
            window_ms=_W_60M,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=0.0,
        ),
        _f_realized_vol_60m_cc,
    ),
    (
        FeatureSpec(
            name=F_VOL_RATIO_15M_60M,
            description="Fifteen-minute volatility divided by sixty-minute volatility.",
            family=FeatureFamily.VOLATILITY,
            source=FeatureSource.CANDLES,
            window_ms=_W_60M,
            market_types=_BOTH,
            fill_value=1.0,
            clip_low=0.0,
            clip_high=10.0,
        ),
        _f_vol_ratio_15m_60m,
    ),
    (
        FeatureSpec(
            name=F_RANGE_VOL_15M_CC,
            description="Additive Parkinson range estimator over fifteen minutes, in centicents.",
            family=FeatureFamily.VOLATILITY,
            source=FeatureSource.CANDLES,
            window_ms=_W_15M,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=0.0,
        ),
        _f_range_vol_15m_cc,
    ),
    # -- flow ----------------------------------------------------------------
    (
        FeatureSpec(
            name=F_VOLUME_5M_QC,
            description="Contracts traded in the last five minutes, in quantity centis.",
            family=FeatureFamily.FLOW,
            source=FeatureSource.CANDLES,
            window_ms=_W_5M,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=0.0,
        ),
        _f_volume_5m_qc,
    ),
    (
        FeatureSpec(
            name=F_VOLUME_ZSCORE_5M,
            description="Five-minute volume standardised against this market's last two hours.",
            family=FeatureFamily.FLOW,
            source=FeatureSource.CANDLES,
            window_ms=_W_120M,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=-10.0,
            clip_high=10.0,
        ),
        _f_volume_zscore_5m,
    ),
    (
        FeatureSpec(
            name=F_TRADE_IMBALANCE_5M,
            description="Signed taker volume over total volume in the last five minutes.",
            family=FeatureFamily.FLOW,
            source=FeatureSource.TICKER,
            window_ms=_W_5M,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=-1.0,
            clip_high=1.0,
        ),
        _f_trade_imbalance_5m,
    ),
    (
        FeatureSpec(
            name=F_OI_CHANGE_5M_QC,
            description="Change in open interest over five minutes, in quantity centis.",
            family=FeatureFamily.FLOW,
            source=FeatureSource.CANDLES,
            window_ms=_W_5M,
            market_types=_BOTH,
            fill_value=0.0,
        ),
        _f_oi_change_5m_qc,
    ),
    (
        FeatureSpec(
            name=F_OI_CHANGE_RATE_5M,
            description="Open-interest change over five minutes as a fraction of the prior level.",
            family=FeatureFamily.FLOW,
            source=FeatureSource.CANDLES,
            window_ms=_W_5M,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=-1.0,
            clip_high=10.0,
        ),
        _f_oi_change_rate_5m,
    ),
    # -- level ---------------------------------------------------------------
    (
        FeatureSpec(
            name=F_PRICE_LEVEL,
            description="Midpoint as an implied probability in [0, 1].",
            family=FeatureFamily.LEVEL,
            source=FeatureSource.BOOK,
            market_types=_EVENT_ONLY,
            clip_low=0.0,
            clip_high=1.0,
        ),
        _f_price_level,
    ),
    (
        FeatureSpec(
            name=F_DISTANCE_FROM_HALF,
            description="Absolute distance of the implied probability from 0.50.",
            family=FeatureFamily.LEVEL,
            source=FeatureSource.BOOK,
            market_types=_EVENT_ONLY,
            clip_low=0.0,
            clip_high=0.5,
        ),
        _f_distance_from_half,
    ),
    (
        FeatureSpec(
            name=F_RANGE_POSITION_60M,
            description="Position of the last close within the sixty-minute high-low range.",
            family=FeatureFamily.LEVEL,
            source=FeatureSource.CANDLES,
            window_ms=_W_60M,
            market_types=_BOTH,
            fill_value=0.5,
            clip_low=0.0,
            clip_high=1.0,
        ),
        _f_range_position_60m,
    ),
    (
        FeatureSpec(
            name=F_RANGE_WIDTH_60M_CC,
            description="Sixty-minute high minus low, in centicents.",
            family=FeatureFamily.LEVEL,
            source=FeatureSource.CANDLES,
            window_ms=_W_60M,
            market_types=_BOTH,
            fill_value=0.0,
            clip_low=0.0,
        ),
        _f_range_width_60m_cc,
    ),
    # -- term ----------------------------------------------------------------
    (
        FeatureSpec(
            name=F_TIME_TO_EXPIRY_MS,
            description="Milliseconds until the market stops trading. Required: no default is honest.",
            family=FeatureFamily.TERM,
            source=FeatureSource.CLOCK,
            market_types=_EVENT_ONLY,
            clip_low=0.0,
        ),
        _f_time_to_expiry_ms,
    ),
    (
        FeatureSpec(
            name=F_LOG_TIME_TO_EXPIRY_H,
            description="log1p of hours until the market stops trading.",
            family=FeatureFamily.TERM,
            source=FeatureSource.CLOCK,
            market_types=_EVENT_ONLY,
            clip_low=0.0,
        ),
        _f_log_time_to_expiry_h,
    ),
    (
        FeatureSpec(
            name=F_EXPIRY_SCALED_MOVE_CC,
            description=(
                "Fifteen-minute volatility scaled by the square root of minutes remaining. "
                "Sourced from candles: the clock component is not an observation and adds no lag."
            ),
            family=FeatureFamily.TERM,
            source=FeatureSource.CANDLES,
            window_ms=_W_15M,
            market_types=_EVENT_ONLY,
            fill_value=0.0,
            clip_low=0.0,
        ),
        _f_expiry_scaled_move_cc,
    ),
    # -- carry ---------------------------------------------------------------
    (
        FeatureSpec(
            name=F_FUNDING_RATE_BPS_HOURLY,
            description="Hourly funding rate in basis points, signed from the long's perspective.",
            family=FeatureFamily.CARRY,
            source=FeatureSource.FUNDING,
            market_types=_PERP_ONLY,
            fill_value=0.0,
            clip_low=-10_000.0,
            clip_high=10_000.0,
        ),
        _f_funding_rate_bps_hourly,
    ),
    (
        FeatureSpec(
            name=F_HOURS_TO_FUNDING,
            description="Hours until the next funding payment.",
            family=FeatureFamily.CARRY,
            source=FeatureSource.FUNDING,
            market_types=_PERP_ONLY,
            fill_value=0.0,
            clip_low=0.0,
            clip_high=24.0,
        ),
        _f_hours_to_funding,
    ),
    (
        FeatureSpec(
            name=F_FUNDING_CARRY_8H_CC,
            description="Funding expected over eight hours, converted into centicents of price.",
            family=FeatureFamily.CARRY,
            source=FeatureSource.FUNDING,
            market_types=_PERP_ONLY,
            fill_value=0.0,
        ),
        _f_funding_carry_8h_cc,
    ),
    (
        FeatureSpec(
            name=F_BASIS_CC,
            description="Exchange mark price minus book midpoint, in centicents.",
            family=FeatureFamily.CARRY,
            source=FeatureSource.TICKER,
            market_types=_PERP_ONLY,
            fill_value=0.0,
        ),
        _f_basis_cc,
    ),
)

ALL_SPECS: Final[tuple[FeatureSpec, ...]] = tuple(spec for spec, _ in _CATALOGUE)
_IMPL: Final[dict[str, _Feature]] = {spec.name: fn for spec, fn in _CATALOGUE}
_SPEC_BY_NAME: Final[dict[str, FeatureSpec]] = {spec.name: spec for spec in ALL_SPECS}


def spec_by_name(name: str) -> FeatureSpec:
    spec = _SPEC_BY_NAME.get(name)
    if spec is None:
        raise DataQualityError(f"unknown feature {name!r}")
    return spec


def event_feature_set(version: str = "v1", *, git_commit: str | None = None) -> FeatureSetSpec:
    """The default event-contract feature set: everything defined for events."""
    return FeatureSetSpec(
        name="event_core",
        version=version,
        market_type=MarketType.EVENT,
        specs=tuple(s for s in ALL_SPECS if s.applies_to(MarketType.EVENT)),
        description="Microstructure, momentum, volatility, flow, level and term features for event contracts.",
        git_commit=git_commit,
    )


def perp_feature_set(version: str = "v1", *, git_commit: str | None = None) -> FeatureSetSpec:
    """The default perpetual feature set: everything defined for perps."""
    return FeatureSetSpec(
        name="perp_core",
        version=version,
        market_type=MarketType.PERP,
        specs=tuple(s for s in ALL_SPECS if s.applies_to(MarketType.PERP)),
        description="Microstructure, momentum, volatility, flow, level and carry features for perpetuals.",
        git_commit=git_commit,
    )


EVENT_FEATURE_SET: Final[FeatureSetSpec] = event_feature_set()
PERP_FEATURE_SET: Final[FeatureSetSpec] = perp_feature_set()


# ---------------------------------------------------------------------------
# Computer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Stats:
    computed: int = 0
    imputed: int = 0
    names_imputed: list[str] = field(default_factory=list)


class FeatureComputer:
    """Turns inputs into a :class:`FeatureVector` for one feature set.

    The same instance serves the training pipeline and the live engine. It holds
    no per-market state, so it is safe to share across markets and across
    decision times; everything mutable lives in the per-call :class:`_Calc`.
    """

    __slots__ = ("_impls", "_specs", "spec")

    def __init__(self, spec: FeatureSetSpec) -> None:
        missing = [s.name for s in spec.specs if s.name not in _IMPL]
        if missing:
            raise DataQualityError(f"no implementation for feature(s): {', '.join(sorted(missing))}")
        self.spec = spec
        self._specs: tuple[FeatureSpec, ...] = spec.specs
        self._impls: tuple[_Feature, ...] = tuple(_IMPL[s.name] for s in spec.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return self.spec.names

    def compute(
        self,
        inputs: FeatureInputs,
        *,
        as_of_ms: int,
        strict: bool = True,
    ) -> FeatureVector:
        """Compute the whole set at ``as_of_ms``.

        ``strict`` runs :meth:`FeatureVector.require_point_in_time` before
        returning. It defaults to on everywhere, including live, because the
        check is microseconds of arithmetic and the failure it catches is a model
        trading on data it never had. Turning it off is only defensible when
        deliberately reproducing a historical vector whose lags have since
        changed.
        """
        if inputs.market_type is not self.spec.market_type:
            raise DataQualityError(
                f"feature set {self.spec.feature_set_id} is for {self.spec.market_type.value} "
                f"markets but was given a {inputs.market_type.value} market ({inputs.ticker})"
            )

        calc = _Calc(inputs, as_of_ms)
        values: dict[str, FeatureValue] = {}
        stats = _Stats()

        with metrics.timer("features.compute_latency_ms", market_type=self.spec.market_type.value):
            for spec, impl in zip(self._specs, self._impls, strict=True):
                result = impl(calc)
                if result is None:
                    if spec.required:
                        raise DataQualityError(
                            f"required feature {spec.name!r} could not be computed for "
                            f"{inputs.ticker} at {as_of_ms}"
                        )
                    assert spec.fill_value is not None
                    values[spec.name] = FeatureValue(
                        name=spec.name,
                        value=spec.fill_value,
                        # An imputed value depends on no observation, so it is
                        # attributed to the decision time itself. Attributing it
                        # to a stale timestamp would make a constant look like
                        # old data and could fail an availability check that has
                        # nothing to check.
                        info_ts_ms=as_of_ms,
                        availability_lag_ms=0,
                        imputed=True,
                    )
                    stats.imputed += 1
                    stats.names_imputed.append(spec.name)
                    continue
                raw, info_ts_ms = result
                values[spec.name] = FeatureValue(
                    name=spec.name,
                    value=spec.sanitise(raw),
                    info_ts_ms=info_ts_ms,
                    availability_lag_ms=spec.availability_lag_ms,
                )
                stats.computed += 1

        vector = FeatureVector(
            ticker=inputs.ticker,
            market_type=inputs.market_type,
            as_of_ms=as_of_ms,
            feature_set_id=self.spec.feature_set_id,
            values=values,
            imputed=tuple(stats.names_imputed),
        )
        if strict:
            vector.require_point_in_time()
        metrics.gauge(
            "features.imputed_fraction",
            stats.imputed / max(1, stats.imputed + stats.computed),
            market_type=self.spec.market_type.value,
        )
        return vector

    def compute_many(
        self,
        inputs: Sequence[FeatureInputs],
        *,
        as_of_ms: int,
        strict: bool = True,
    ) -> list[FeatureVector]:
        return [self.compute(i, as_of_ms=as_of_ms, strict=strict) for i in inputs]


# Referenced in the module docstring's contract with the strategy layer; kept
# here so a reader can see at a glance which lags the catalogue actually uses.
_LAGS_IN_USE: Final[tuple[int, ...]] = (
    LAG_BOOK_MS,
    LAG_CANDLE_MS,
    LAG_TICKER_MS,
    LAG_FUNDING_MS,
    LAG_CLOCK_MS,
)
