"""Shared arithmetic and plumbing for concrete strategies.

Every strategy in this package answers the same question -- "is this trade worth
doing after costs?" -- and that question is the easiest one in the platform to
get wrong in the optimistic direction. Three mistakes account for almost all of
it: counting only the entry fee (which roughly doubles an apparent edge), pricing
a fill at the touch when the order will walk three levels, and rounding a
modelled fair value away from the market so that arithmetic manufactures edge.

This module exists so those mistakes are made once, correctly, rather than in
eight strategies, differently:

* :func:`event_costing` and :func:`perp_costing` always charge a **round trip**
  and always work in integers, never floats.
* :func:`event_execution_cost` and :func:`perp_execution_cost` price a fill by
  walking the book for the size actually requested, so the number used is the
  cost of the whole order rather than the cost of its first contract.
* :func:`conservative_fair_value_cc` rounds a modelled fair value **toward** the
  market, so a fraction of a centicent can never become an edge.

Feature names are constants rather than string literals because they are a
contract with the feature engine: a strategy declares what it needs through
``requires_features`` and the engine refuses to run it when it cannot supply
them. A typo in a literal would silently become a missing feature defaulting to
zero, which is precisely how a strategy ends up trading on nothing.

Where floats are allowed
------------------------
Probabilities, volatilities, correlations and decay weights are floats -- they
are statistics and have no exact decimal representation to preserve. Every
quantity denominated in money or in contracts is an integer at the platform
scales, and the crossing point between the two is
:func:`conservative_fair_value_cc`, which is deliberately the only place a float
becomes a price.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal
from typing import Any, ClassVar

from kat.common.enums import BookSide, MarketType, OutcomeSide, StrategyState
from kat.common.errors import ConfigError
from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT, Rounding, parse_dollars_micros
from kat.kalshi.fees import FeeSchedule, round_trip_fee_micros
from kat.marketdata.book import OrderBook
from kat.risk.types import OrderIntent
from kat.strategies.base import Strategy, StrategyContext, StrategyMetadata, StrategyParam

__all__ = [
    "CAPITAL_PARAM_SPACE",
    "F_BOOK_IMBALANCE",
    "F_MID_CHANGE_1M_CC",
    "F_MID_CHANGE_5M_CC",
    "F_MID_CHANGE_15M_CC",
    "F_MODEL_CONFIDENCE",
    "F_REALIZED_VOL_5M_CC",
    "F_TIME_TO_EXPIRY_MS",
    "F_TRADE_IMBALANCE_5M",
    "F_VOLUME_5M_QC",
    "F_VOLUME_ZSCORE_5M",
    "MIN_ORDER_QC",
    "PRICE_MAX_CC",
    "PRICE_MIN_CC",
    "Costing",
    "EventQuote",
    "ExecutionCost",
    "PriceTracker",
    "TradingStrategy",
    "clamp_price_cc",
    "conservative_fair_value_cc",
    "contracts_for_budget_qc",
    "event_costing",
    "event_execution_cost",
    "fee_schedule_for",
    "funding_benefit_micros",
    "perp_costing",
    "perp_execution_cost",
    "probability_to_cc",
    "resolve_params",
    "time_to_expiry_ms",
]

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
# Event-contract features. Price changes are *additive* in centicents rather
# than multiplicative in basis points: a market moving from $0.05 to $0.10 has
# doubled, but what matters for an event contract is that the implied
# probability rose by five points, and a percentage return would rank that move
# far above an identical five-point move from $0.50.
F_MID_CHANGE_1M_CC = "mid_change_1m_cc"
F_MID_CHANGE_5M_CC = "mid_change_5m_cc"
F_MID_CHANGE_15M_CC = "mid_change_15m_cc"
#: Standard deviation of one-minute mid changes over the last 15 minutes, in cc.
F_REALIZED_VOL_5M_CC = "realized_vol_5m_cc"
F_VOLUME_5M_QC = "volume_5m_qc"
#: Standardised five-minute volume against the market's own trailing baseline.
F_VOLUME_ZSCORE_5M = "volume_zscore_5m"
#: Signed buy/sell pressure from trade prints in [-1, 1]; +1 is all lifting.
F_TRADE_IMBALANCE_5M = "trade_imbalance_5m"
#: (bid_depth - ask_depth) / (bid_depth + ask_depth) over the top levels.
F_BOOK_IMBALANCE = "book_imbalance"
#: Milliseconds until the market stops trading. Float only because the feature
#: bus is float-valued; it is an integer quantity and is used as one.
F_TIME_TO_EXPIRY_MS = "time_to_expiry_ms"
#: Model self-reported reliability for this market in [0, 1], when available.
F_MODEL_CONFIDENCE = "model_confidence"

PRICE_MIN_CC = 1
PRICE_MAX_CC = CC_PER_DOLLAR - 1
#: Kalshi trades whole contracts; a sub-contract intent is not expressible.
MIN_ORDER_QC = QC_PER_CONTRACT

_MS_PER_HOUR = 3_600_000.0


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def clamp_price_cc(value: int) -> int:
    """Clamp a price onto the tradeable grid.

    $0.00 and $1.00 are not prices, they are settlements. A model that produces
    one of them is saying it is certain, and acting on certainty is how a
    strategy sizes a position it cannot exit.
    """
    return max(PRICE_MIN_CC, min(PRICE_MAX_CC, int(value)))


def probability_to_cc(probability: float) -> int:
    """Convert a probability in [0, 1] into a centicent price."""
    return clamp_price_cc(round(probability * CC_PER_DOLLAR))


def conservative_fair_value_cc(raw_cc: float, market_cc: int) -> int:
    """Round a modelled fair value toward the market price.

    The direction of the rounding is the whole point. Rounding away from the
    market would let sub-centicent modelling noise become a fraction of a
    centicent of apparent edge on every contract, which across thousands of
    contracts is real money that never existed.
    """
    if raw_cc >= market_cc:
        return clamp_price_cc(math.floor(raw_cc))
    return clamp_price_cc(math.ceil(raw_cc))


def fee_schedule_for(ctx: StrategyContext) -> FeeSchedule:
    """Fee schedule implied by the context.

    When the multiplier is unknown the schedule is marked inexact, which makes
    :func:`kat.kalshi.fees.estimate_fee_micros` add a safety margin. A strategy
    therefore trades *less* when we do not know what the exchange will charge,
    which is the correct response to that uncertainty.
    """
    return FeeSchedule.from_series("quadratic", ctx.fee_multiplier)


def time_to_expiry_ms(ctx: StrategyContext) -> int | None:
    """Milliseconds until this market stops trading, or ``None`` if unknown.

    Prefers the feature, which the feature engine computes from the market
    definition, and falls back to a close timestamp carried on the snapshot
    metadata. Returns ``None`` rather than guessing: a strategy gated on time to
    expiry must not run when it does not know the time to expiry.
    """
    raw = ctx.features.get(F_TIME_TO_EXPIRY_MS)
    if raw is not None and math.isfinite(raw):
        return int(raw)
    close_ms = ctx.market.metadata.get("close_ts_ms")
    if isinstance(close_ms, int):
        return close_ms - ctx.now_ms
    return None


# ---------------------------------------------------------------------------
# Event-contract quote view
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventQuote:
    """The two prices an event-contract strategy can actually trade at.

    The book is YES-denominated (see :mod:`kat.marketdata.book`), so buying NO
    is not "selling the ask" -- it is trading against the resting YES *bids*, at
    a cost of ``$1.00 - yes_bid``. Keeping that conversion in one place is the
    difference between a strategy that is short when it thinks it is long and
    one that is not.
    """

    ticker: str
    yes_bid_cc: int
    yes_ask_cc: int
    yes_bid_size_qc: int
    yes_ask_size_qc: int
    bid_depth_qc: int
    ask_depth_qc: int

    @property
    def mid_cc(self) -> int:
        return (self.yes_bid_cc + self.yes_ask_cc) // 2

    @property
    def spread_cc(self) -> int:
        return self.yes_ask_cc - self.yes_bid_cc

    @property
    def no_ask_cc(self) -> int:
        """What one NO contract costs to buy."""
        return CC_PER_DOLLAR - self.yes_bid_cc

    def cost_cc(self, side: OutcomeSide) -> int:
        """Touch cost of buying one contract of ``side``."""
        return self.yes_ask_cc if side is OutcomeSide.YES else self.no_ask_cc

    def touch_size_qc(self, side: OutcomeSide) -> int:
        """Size available at the touch for a buy of ``side``."""
        return self.yes_ask_size_qc if side is OutcomeSide.YES else self.yes_bid_size_qc

    @classmethod
    def from_book(cls, book: OrderBook | None) -> EventQuote | None:
        """Build a quote, or ``None`` when the book is unusable.

        A one-sided or crossed book yields ``None``. Both conditions mean the
        market cannot be priced, and inventing a price from the last trade is
        how a strategy ends up quoting against a market that is not there.
        """
        if book is None or book.is_crossed:
            return None
        bid, ask = book.best_bid_cc, book.best_ask_cc
        if bid is None or ask is None:
            return None
        return cls(
            ticker=book.ticker,
            yes_bid_cc=bid,
            yes_ask_cc=ask,
            yes_bid_size_qc=book.best_bid_size_qc,
            yes_ask_size_qc=book.best_ask_size_qc,
            bid_depth_qc=book.depth_qc(is_bid=True, levels=5),
            ask_depth_qc=book.depth_qc(is_bid=False, levels=5),
        )


# ---------------------------------------------------------------------------
# Execution cost
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionCost:
    """What an order of a given size would actually cost to fill.

    ``vwap_cc`` is the average price across every level consumed and is what
    expected value is computed against. ``worst_cc`` is the last level touched
    and is what the limit price is set to, so the whole order can fill. They are
    different numbers and using the first where the second belongs produces
    orders that fill half.
    """

    qty_qc: int
    vwap_cc: int
    worst_cc: int


def _fillable_qc(book: OrderBook, *, is_bid_side: bool, levels: int, wanted_qc: int) -> int:
    """Largest whole-contract size available within ``levels`` of the book."""
    depth = book.depth_qc(is_bid=is_bid_side, levels=levels)
    usable = min(wanted_qc, depth)
    return (usable // QC_PER_CONTRACT) * QC_PER_CONTRACT


def event_execution_cost(
    book: OrderBook, side: OutcomeSide, wanted_qc: int, *, levels: int = 3
) -> ExecutionCost | None:
    """Cost of buying up to ``wanted_qc`` of ``side``, walking at most ``levels``.

    Capping the walk is a market-impact control, not a convenience: an order
    that reaches the fifth level of a Kalshi event book has moved the market it
    is trying to trade, and the price it gets is no longer the price the signal
    was measured at.
    """
    buying_yes = side is OutcomeSide.YES
    # Buying YES consumes the ask side; buying NO consumes the resting YES bids.
    consumes_bids = not buying_yes
    qty = _fillable_qc(book, is_bid_side=consumes_bids, levels=levels, wanted_qc=wanted_qc)
    if qty < MIN_ORDER_QC:
        return None

    walked = book.walk(qty, is_buy=buying_yes)
    if walked is None:
        return None
    filled_qc, total_micros, worst_cc = walked
    if filled_qc < qty:
        return None

    if buying_yes:
        # Round the average we pay up; it is a cost.
        vwap = -(-total_micros // filled_qc)
        return ExecutionCost(qty_qc=qty, vwap_cc=clamp_price_cc(vwap), worst_cc=clamp_price_cc(worst_cc))

    # NO cost is the reflection of the YES bids consumed. Flooring the YES side
    # before reflecting rounds the NO cost up, which is again against us.
    yes_vwap = total_micros // filled_qc
    return ExecutionCost(
        qty_qc=qty,
        vwap_cc=clamp_price_cc(CC_PER_DOLLAR - yes_vwap),
        worst_cc=clamp_price_cc(CC_PER_DOLLAR - worst_cc),
    )


def perp_execution_cost(
    book: OrderBook, *, is_buy: bool, wanted_qc: int, levels: int = 3
) -> ExecutionCost | None:
    """Cost of a marketable perpetual order of up to ``wanted_qc``."""
    qty = _fillable_qc(book, is_bid_side=not is_buy, levels=levels, wanted_qc=wanted_qc)
    if qty < MIN_ORDER_QC:
        return None
    walked = book.walk(qty, is_buy=is_buy)
    if walked is None:
        return None
    filled_qc, total_micros, worst_cc = walked
    if filled_qc < qty:
        return None
    # A buyer's average rounds up, a seller's rounds down: both against us.
    vwap = -(-total_micros // filled_qc) if is_buy else total_micros // filled_qc
    return ExecutionCost(qty_qc=qty, vwap_cc=clamp_price_cc(vwap), worst_cc=clamp_price_cc(worst_cc))


# ---------------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Costing:
    """The full expected-value arithmetic for one candidate trade.

    Every field is an exact integer and every field is recorded on the intent's
    metadata, so a decision can be re-derived later without re-running the
    strategy.
    """

    qty_qc: int
    cost_cc: int
    value_cc: int
    edge_cc: int
    gross_micros: int
    fees_micros: int
    funding_micros: int
    net_micros: int

    @property
    def notional_micros(self) -> int:
        return self.cost_cc * self.qty_qc

    @property
    def is_profitable(self) -> bool:
        return self.net_micros > 0


def event_costing(
    *,
    side: OutcomeSide,
    cost_cc: int,
    fair_value_cc: int,
    qty_qc: int,
    schedule: FeeSchedule,
) -> Costing:
    """Expected value of buying ``qty_qc`` of ``side`` at ``cost_cc``.

    ``fair_value_cc`` is always YES-denominated, matching what the risk engine
    expects on an intent; the NO value is derived here rather than at each call
    site. The exit leg of the fee is charged at the fair value, on the
    assumption that the position is closed rather than held to settlement. That
    is the conservative assumption: settlement is free on most Kalshi series, so
    charging an exit fee can only understate the edge.
    """
    value_cc = fair_value_cc if side is OutcomeSide.YES else CC_PER_DOLLAR - fair_value_cc
    edge_cc = value_cc - cost_cc
    gross_micros = edge_cc * qty_qc
    fees_micros = round_trip_fee_micros(
        entry_price_cc=cost_cc,
        exit_price_cc=clamp_price_cc(value_cc),
        qty_qc=qty_qc,
        schedule=schedule,
    )
    return Costing(
        qty_qc=qty_qc,
        cost_cc=cost_cc,
        value_cc=value_cc,
        edge_cc=edge_cc,
        gross_micros=gross_micros,
        fees_micros=fees_micros,
        funding_micros=0,
        net_micros=gross_micros - fees_micros,
    )


def perp_costing(
    *,
    is_long: bool,
    entry_cc: int,
    fair_cc: int,
    qty_qc: int,
    schedule: FeeSchedule,
    funding_micros: int = 0,
) -> Costing:
    """Expected value of a perpetual position taken at ``entry_cc``.

    ``funding_micros`` is signed and is what funding would *pay us* over the
    intended holding period -- negative when we are the side that pays. It is
    part of the edge on a perpetual in a way it never is on an event contract,
    so it is an input here rather than a separate check.
    """
    edge_cc = (fair_cc - entry_cc) if is_long else (entry_cc - fair_cc)
    gross_micros = edge_cc * qty_qc
    fees_micros = round_trip_fee_micros(
        entry_price_cc=entry_cc,
        exit_price_cc=clamp_price_cc(fair_cc),
        qty_qc=qty_qc,
        schedule=schedule,
    )
    return Costing(
        qty_qc=qty_qc,
        cost_cc=entry_cc,
        value_cc=fair_cc,
        edge_cc=edge_cc,
        gross_micros=gross_micros,
        fees_micros=fees_micros,
        funding_micros=funding_micros,
        net_micros=gross_micros + funding_micros - fees_micros,
    )


def funding_benefit_micros(
    *, rate_bps_per_hour: float, notional_micros: int, hours: float, is_long: bool
) -> int:
    """Funding a position of this notional would receive over ``hours``.

    Sign convention follows the exchange: a positive rate means longs pay
    shorts, so the result is negative for a long in positive funding. The value
    is floored rather than rounded, which shrinks a receipt and enlarges a
    payment -- conservative in both directions at once.
    """
    if notional_micros <= 0 or hours <= 0:
        return 0
    rate = Decimal(str(rate_bps_per_hour)) / Decimal(10_000)
    magnitude = Decimal(notional_micros) * rate * Decimal(str(hours))
    signed = -magnitude if is_long else magnitude
    return int(signed.to_integral_value(rounding=ROUND_FLOOR))


def contracts_for_budget_qc(*, budget_micros: int, price_cc: int) -> int:
    """Whole contracts affordable at ``price_cc``, expressed in centi-contracts."""
    if budget_micros <= 0 or price_cc <= 0:
        return 0
    contracts = budget_micros // (price_cc * QC_PER_CONTRACT)
    return int(contracts) * QC_PER_CONTRACT


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

#: Sizing parameters every strategy carries. They are part of the search space
#: because position size is a strategy parameter like any other -- but the risk
#: engine, not this number, decides what is actually permitted.
CAPITAL_PARAM_SPACE: tuple[StrategyParam, ...] = (
    StrategyParam(
        "max_notional_usd", 25.0, low=1.0, high=500.0, step=1.0,
        description="hard ceiling on the notional of one intent, in dollars",
    ),
    StrategyParam(
        "capital_fraction_bps", 2500, low=100, high=10_000, step=100,
        description="fraction of allocated capital one intent may request, in basis points",
    ),
    StrategyParam(
        "max_book_levels", 3, low=1, high=5, step=1,
        description="how many book levels an order may walk before it is considered impactful",
    ),
)


def resolve_params(
    space: tuple[StrategyParam, ...], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Merge caller overrides onto the declared defaults, clamped to the space.

    An unknown key raises rather than being ignored. The optimiser writes these
    dictionaries, and silently dropping a parameter it thought it had set would
    make an entire search invalid in a way nothing would ever surface.
    """
    resolved: dict[str, Any] = {p.name: p.default for p in space}
    if not overrides:
        return resolved
    by_name = {p.name: p for p in space}
    unknown = sorted(set(overrides) - set(by_name))
    if unknown:
        raise ConfigError(f"unknown strategy parameters: {', '.join(unknown)}")
    for name, value in overrides.items():
        resolved[name] = by_name[name].clamp(value)
    return resolved


# ---------------------------------------------------------------------------
# Mark tracking (perpetuals)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MarkSeries:
    """Exponentially weighted view of one market's price history."""

    first_ms: int
    last_ms: int
    last_price_cc: int
    ema_fast: float
    ema_slow: float
    # EWMA of squared log return per millisecond -- a diffusion variance rate,
    # so that irregular sampling does not change the estimate's scale.
    var_per_ms: float = 0.0
    samples: int = 1


class PriceTracker:
    """Time-decayed EMAs and realised volatility, driven by the platform clock.

    Perpetual marks tick continuously and irregularly. A fixed-window moving
    average over "the last N updates" therefore measures a different amount of
    time depending on how busy the market is, which makes a parameter tuned in a
    quiet hour wrong in a busy one. The weights here decay with *elapsed time*,
    so a parameter means the same thing at every level of activity.

    This lives in the strategy rather than in the feature engine because the
    time constants are strategy parameters the optimiser searches; publishing
    them as features would require the feature engine to know each strategy's
    current parameters, which is a coupling with no benefit.
    """

    __slots__ = ("_fast_tau_ms", "_max_tickers", "_series", "_slow_tau_ms", "_vol_tau_ms")

    def __init__(
        self,
        *,
        fast_tau_ms: int,
        slow_tau_ms: int,
        vol_tau_ms: int = 1_800_000,
        max_tickers: int = 512,
    ) -> None:
        self._fast_tau_ms = max(1, fast_tau_ms)
        self._slow_tau_ms = max(1, slow_tau_ms)
        self._vol_tau_ms = max(1, vol_tau_ms)
        self._max_tickers = max(1, max_tickers)
        self._series: dict[str, _MarkSeries] = {}

    def update(self, ticker: str, price_cc: int, now_ms: int) -> _MarkSeries | None:
        """Fold one observation in and return the updated series.

        Out-of-order and duplicate timestamps are ignored rather than folded
        with a negative or zero interval: replaying a backtest through a
        strategy that treats them as new information gives different answers on
        a rerun.
        """
        if price_cc <= 0:
            return None
        series = self._series.get(ticker)
        if series is None:
            self._evict_if_full()
            series = _MarkSeries(
                first_ms=now_ms,
                last_ms=now_ms,
                last_price_cc=price_cc,
                ema_fast=float(price_cc),
                ema_slow=float(price_cc),
            )
            self._series[ticker] = series
            return series

        dt_ms = now_ms - series.last_ms
        if dt_ms <= 0:
            return series

        price = float(price_cc)
        series.ema_fast += (1.0 - math.exp(-dt_ms / self._fast_tau_ms)) * (price - series.ema_fast)
        series.ema_slow += (1.0 - math.exp(-dt_ms / self._slow_tau_ms)) * (price - series.ema_slow)

        log_return = math.log(price / series.last_price_cc)
        inst_var = (log_return * log_return) / dt_ms
        weight = 1.0 - math.exp(-dt_ms / self._vol_tau_ms)
        series.var_per_ms += weight * (inst_var - series.var_per_ms)

        series.last_ms = now_ms
        series.last_price_cc = price_cc
        series.samples += 1
        return series

    def get(self, ticker: str) -> _MarkSeries | None:
        return self._series.get(ticker)

    def observed_ms(self, ticker: str) -> int:
        series = self._series.get(ticker)
        return 0 if series is None else series.last_ms - series.first_ms

    def vol_bps_per_hour(self, ticker: str) -> float | None:
        """Realised volatility over a one-hour horizon, in basis points.

        ``None`` until enough observations exist for the estimate to mean
        anything; a volatility computed from two ticks is a number, not an
        estimate.
        """
        series = self._series.get(ticker)
        if series is None or series.samples < 8 or series.var_per_ms <= 0.0:
            return None
        return math.sqrt(series.var_per_ms * _MS_PER_HOUR) * 10_000.0

    def forget(self, ticker: str) -> None:
        self._series.pop(ticker, None)

    def _evict_if_full(self) -> None:
        """Drop the least recently updated series once the table is full.

        Unbounded per-ticker state in a long-lived process is a slow leak. The
        eviction is cheap because it happens only on the first sighting of a new
        ticker after the table has filled.
        """
        if len(self._series) < self._max_tickers:
            return
        oldest = min(self._series, key=lambda t: self._series[t].last_ms)
        del self._series[oldest]


# ---------------------------------------------------------------------------
# Strategy base
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SkipLog:
    """Why a strategy declined to trade, and how often.

    Refusal is the normal outcome, so the reasons for it are the primary
    diagnostic: a strategy producing nothing because its market is illiquid is
    healthy, and one producing nothing because a feature it needs is never
    supplied is broken. Without this they look identical from outside.
    """

    counts: dict[str, int] = field(default_factory=dict)
    last: str | None = None

    def record(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        self.last = reason


class TradingStrategy(Strategy):
    """Construction, sizing and diagnostics shared by every concrete strategy.

    Subclasses declare their identity as class attributes and implement
    :meth:`~kat.strategies.base.Strategy.generate`. They are expected to call
    :meth:`skip` on every path that declines to trade, which costs nothing and
    turns "this strategy has never traded" from a mystery into a report.
    """

    FAMILY: ClassVar[str] = ""
    NAME: ClassVar[str] = ""
    VERSION: ClassVar[str] = "1.0.0"
    MARKET_TYPE: ClassVar[MarketType] = MarketType.EVENT
    DESCRIPTION: ClassVar[str] = ""
    PARAM_SPACE: ClassVar[tuple[StrategyParam, ...]] = ()
    REQUIRES_FEATURES: ClassVar[tuple[str, ...]] = ()
    REQUIRES_BOOK: ClassVar[bool] = True
    MAX_DATA_AGE_MS: ClassVar[int] = 5_000

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        strategy_id: str | None = None,
        version: str | None = None,
        state: StrategyState = StrategyState.DEVELOPMENT,
        discovered_by: str = "human",
        git_commit: str | None = None,
    ) -> None:
        space = tuple(CAPITAL_PARAM_SPACE) + tuple(self.PARAM_SPACE)
        metadata = StrategyMetadata(
            strategy_id=strategy_id or f"{self.MARKET_TYPE.value}.{self.FAMILY}",
            version=version or self.VERSION,
            name=self.NAME or type(self).__name__,
            market_type=self.MARKET_TYPE,
            family=self.FAMILY,
            description=self.DESCRIPTION.strip(),
            params=resolve_params(space, params),
            param_space=space,
            discovered_by=discovered_by,
            git_commit=git_commit,
            requires_book=self.REQUIRES_BOOK,
            requires_features=tuple(self.REQUIRES_FEATURES),
            max_data_age_ms=self.MAX_DATA_AGE_MS,
        )
        super().__init__(metadata, state)
        self._calls = 0
        self._intents = 0
        self._skips = _SkipLog()

    # -- parameters ----------------------------------------------------------

    def int_param(self, name: str) -> int:
        return int(self.params[name])

    def float_param(self, name: str) -> float:
        return float(self.params[name])

    def budget_micros(self) -> int:
        """Capital this strategy is prepared to commit to one new position.

        The smaller of its own configured ceiling and its share of whatever the
        portfolio layer allocated. When nothing has been allocated the ceiling
        applies alone, which is what makes a strategy usable in a backtest that
        has no portfolio layer at all.
        """
        ceiling = parse_dollars_micros(self.params["max_notional_usd"], rounding=Rounding.FLOOR)
        allocated = self.allocated_capital_micros
        if allocated > 0:
            share = allocated * self.int_param("capital_fraction_bps") // 10_000
            ceiling = min(ceiling, share)
        return max(0, ceiling)

    # -- diagnostics ---------------------------------------------------------

    def skip(self, reason: str) -> list[OrderIntent]:
        """Record why no intent was produced and return the empty list."""
        self._skips.record(reason)
        return []

    def emit(self, intents: list[OrderIntent]) -> list[OrderIntent]:
        self._intents += len(intents)
        return intents

    def begin(self, ctx: StrategyContext) -> None:
        self._calls += 1

    def health(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "intents": self._intents,
            "skips": dict(sorted(self._skips.counts.items(), key=lambda kv: -kv[1])),
            "last_skip": self._skips.last,
        }

    # -- intent construction -------------------------------------------------

    def event_intent(
        self,
        ctx: StrategyContext,
        *,
        side: OutcomeSide,
        qty_qc: int,
        limit_price_cc: int,
        fair_value_cc: int,
        confidence: float,
        costing: Costing,
        reason: str,
        extra: Mapping[str, Any] | None = None,
    ) -> OrderIntent:
        """Build an event-contract intent with its arithmetic attached.

        ``fair_value_cc`` on the intent is always the YES-denominated fair
        value, because that is what :class:`kat.risk.event_risk.EventRiskEngine`
        expects; it converts for a NO order itself.
        """
        return OrderIntent(
            market_type=MarketType.EVENT,
            ticker=ctx.market.ticker,
            qty_qc=qty_qc,
            limit_price_cc=limit_price_cc,
            outcome_side=side,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            expected_value_micros=costing.net_micros,
            fair_value_cc=fair_value_cc,
            confidence=round(confidence, 4),
            metadata=self._intent_metadata(ctx, costing, reason, extra),
        )

    def perp_intent(
        self,
        ctx: StrategyContext,
        *,
        is_buy: bool,
        qty_qc: int,
        limit_price_cc: int,
        fair_value_cc: int,
        confidence: float,
        costing: Costing,
        reason: str,
        reduce_only: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> OrderIntent:
        return OrderIntent(
            market_type=MarketType.PERP,
            ticker=ctx.market.ticker,
            qty_qc=qty_qc,
            limit_price_cc=limit_price_cc,
            book_side=BookSide.BID if is_buy else BookSide.ASK,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            reduce_only=reduce_only,
            expected_value_micros=costing.net_micros,
            fair_value_cc=fair_value_cc,
            confidence=round(confidence, 4),
            metadata=self._intent_metadata(ctx, costing, reason, extra),
        )

    def _intent_metadata(
        self,
        ctx: StrategyContext,
        costing: Costing,
        reason: str,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "reason": reason,
            "family": self.FAMILY,
            "edge_cc": costing.edge_cc,
            "gross_ev_micros": costing.gross_micros,
            "fees_micros": costing.fees_micros,
            "net_ev_micros": costing.net_micros,
            "decided_at_ms": ctx.now_ms,
        }
        if costing.funding_micros:
            meta["funding_micros"] = costing.funding_micros
        if ctx.regime:
            meta["regime"] = ctx.regime
        if extra:
            meta.update(extra)
        return meta
