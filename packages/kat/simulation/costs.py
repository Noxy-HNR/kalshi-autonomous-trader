"""What a simulated fill actually costs.

A backtest is a claim about money, and nearly every way a backtest lies is a
cost it forgot to charge. This module is the one place the simulator decides
what a fill costs, so that the four charges that matter are all present and all
visible in the result:

* **Fees** come from :mod:`kat.kalshi.fees`, never from a hand-rolled
  percentage. When the series fee multiplier is unknown that module marks the
  estimate inexact and adds a safety margin, and the simulator keeps that
  pessimism rather than papering over it.
* **Spread** is paid, always. A buy lifts the ask and a sell hits the bid.
  There is no midpoint fill anywhere in this package, because a midpoint fill is
  a free half-spread on every trade and it is the single largest source of
  imaginary backtest profit.
* **Slippage** is measured by walking the real book
  (:meth:`kat.marketdata.book.OrderBook.walk`), so a size larger than the touch
  pays for the levels it consumes rather than pretending the touch was infinite.
* **Latency** is modelled as *elapsed simulated time* between the decision and
  the order's arrival, during which the book is free to move. It is not a fudge
  factor added to the price: the difference between a naive and an honest
  backtest is that the honest one re-reads the book on arrival.

Why the defaults are pessimistic
--------------------------------
The cost model is the parameter a researcher is most tempted to tune, and every
optimistic choice makes a marginal strategy look promotable. The defaults here
are therefore set where an unlucky day would put them, not where an average one
would:

* latency of 250 ms with **one-sided** jitter -- a latency sample is never
  *better* than the configured mean, because a fast day is not a risk;
* a resting order is assumed to sit behind **all** existing size at its price
  (:attr:`CostModel.queue_ahead_bps` = 10000);
* fees default to the inexact schedule, which carries a 1.5x margin.

:data:`FRICTIONLESS` exists so a researcher can measure how much of a result is
cost-sensitive by differencing the two runs. It is a diagnostic, and a run
performed with it is not admissible evidence for promoting anything -- which is
why :class:`~kat.simulation.engine.BacktestResult` records the cost model it
used alongside every number it reports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any

from kat.common.fixedpoint import micros_to_decimal
from kat.kalshi.fees import (
    FeeSchedule,
    PerpFeeSchedule,
    estimate_fee_micros,
    perp_fee_micros,
)
from kat.marketdata.book import OrderBook

__all__ = [
    "FRICTIONLESS",
    "PESSIMISTIC",
    "CostModel",
    "CostReport",
    "LatencyModel",
]

#: Full weight, in basis points, for the "all resting size at our price is ahead
#: of us" queue assumption.
_FULL_BPS = 10_000


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Delay between deciding and the exchange seeing the order.

    Jitter is deliberately **one-sided**: a sample is drawn in
    ``[decision_to_arrival_ms, decision_to_arrival_ms + jitter_ms]``. Two-sided
    jitter would let half of all orders arrive faster than the configured
    latency, and a backtest does not need protection from the possibility that
    the network was quick.

    ``market_data_lag_ms`` is separate from order latency and is applied by the
    feed builder, not here: it describes how late we *learn* about the market,
    whereas the fields below describe how late our reaction *reaches* it.
    """

    decision_to_arrival_ms: int = 250
    jitter_ms: int = 150
    cancel_ms: int = 250
    market_data_lag_ms: int = 0

    def sample_arrival_delay_ms(self, rng: random.Random) -> int:
        base = max(0, self.decision_to_arrival_ms)
        if self.jitter_ms <= 0:
            return base
        return base + int(rng.random() * self.jitter_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_to_arrival_ms": self.decision_to_arrival_ms,
            "jitter_ms": self.jitter_ms,
            "cancel_ms": self.cancel_ms,
            "market_data_lag_ms": self.market_data_lag_ms,
        }


@dataclass(frozen=True, slots=True)
class CostModel:
    """Every charge the simulator applies, in one configurable object.

    Instances are frozen and are recorded verbatim on the backtest result, so a
    number can never be quoted without the cost assumptions that produced it.
    """

    fee_schedule: FeeSchedule = field(default_factory=FeeSchedule)
    #: Rates for a perpetual, when this run is trading one. Perpetuals pay a
    #: flat fraction of notional -- 2bp to a maker, 4bp to a taker -- and not
    #: the quadratic event-contract fee, which at fifty cents is 350bp and
    #: would overstate the cost by about a hundredfold. Left None the
    #: quadratic schedule applies, which is right for event contracts and
    #: wrong for anything else.
    perp_fees: PerpFeeSchedule | None = None
    latency: LatencyModel = field(default_factory=LatencyModel)

    #: Fraction of the size already resting at our price that we assume is
    #: ahead of us in the queue, in basis points. 10000 means "all of it", which
    #: is the correct assumption for a platform that is never the first quote.
    queue_ahead_bps: int = _FULL_BPS

    #: Additional price concession charged on a taker fill, in centicents, on
    #: top of what walking the book already costs. Zero by default because the
    #: book walk is the honest measure; raise it to stress-test a strategy
    #: against a book that is thinner than the one recorded.
    taker_concession_cc: int = 0

    #: How many levels a marketable order may consume. Beyond this the order
    #: partially fills and the remainder rests, rather than sweeping a book that
    #: in reality would have pulled.
    max_walk_levels: int = 5

    #: A resting order is cancelled after this long without filling. ``None``
    #: leaves it working for the whole run, which flatters any strategy whose
    #: edge is really "eventually the market came back".
    order_ttl_ms: int | None = None

    #: Charge maker fills at the taker rate. Kalshi's plain ``quadratic`` series
    #: have no maker discount at all, so this only matters on the maker-fee
    #: variants; leaving it True is the conservative reading.
    charge_maker_as_taker: bool = False

    # -- fees ---------------------------------------------------------------

    def fee_micros(self, *, price_cc: int, qty_qc: int, is_taker: bool) -> int:
        """Fee for one fill.

        For an event contract the price passed here must be the price of the
        contract actually traded -- the NO price for a NO fill. The quadratic
        formula is symmetric about $0.50 so ``f(p) == f(1-p)``, but relying on
        that symmetry instead of passing the right price would break the moment
        a series uses the flat schedule.
        """
        effective_taker = True if self.charge_maker_as_taker else is_taker
        if self.perp_fees is not None:
            return perp_fee_micros(
                price_cc=price_cc,
                qty_qc=qty_qc,
                schedule=self.perp_fees,
                is_taker=effective_taker,
            )
        return estimate_fee_micros(
            price_cc=price_cc,
            qty_qc=qty_qc,
            schedule=self.fee_schedule,
            is_taker=effective_taker,
        ).fee_micros

    # -- book-derived costs --------------------------------------------------

    def slippage_cc(self, book: OrderBook, qty_qc: int, *, is_buy: bool) -> int | None:
        """Distance between the VWAP for this size and the touch, or ``None``.

        ``None`` means the book could not fill the size at all, which is
        information rather than a zero.
        """
        return book.slippage_cc(qty_qc, is_buy=is_buy)

    def fillable_qc(self, book: OrderBook, *, is_buy: bool) -> int:
        """Size available within the levels this model permits an order to walk."""
        return book.depth_qc(is_bid=not is_buy, levels=self.max_walk_levels)

    def queue_ahead_qc(self, resting_qc: int) -> int:
        """Size assumed to be ahead of us at our price when an order rests."""
        if resting_qc <= 0:
            return 0
        return resting_qc * self.queue_ahead_bps // _FULL_BPS

    def apply_taker_concession_cc(self, price_cc: int, *, is_buy: bool) -> int:
        """Move a taker fill price against us by the configured concession."""
        if self.taker_concession_cc <= 0:
            return price_cc
        return price_cc + self.taker_concession_cc if is_buy else price_cc - self.taker_concession_cc

    # -- variants ------------------------------------------------------------

    def with_fee_multiplier(self, multiplier: float | None, fee_type: str = "quadratic") -> CostModel:
        """Copy with the fee schedule replaced by a known series multiplier."""
        return replace(self, fee_schedule=FeeSchedule.from_series(fee_type, multiplier))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fee_type": self.fee_schedule.fee_type,
            "fee_multiplier": str(self.fee_schedule.multiplier),
            "fee_estimate_exact": self.fee_schedule.exact,
            "latency": self.latency.to_dict(),
            "queue_ahead_bps": self.queue_ahead_bps,
            "perp_fees": (
                {
                    "maker_rate": str(self.perp_fees.maker_rate),
                    "taker_rate": str(self.perp_fees.taker_rate),
                    "exact": self.perp_fees.exact,
                }
                if self.perp_fees is not None
                else None
            ),
            "taker_concession_cc": self.taker_concession_cc,
            "max_walk_levels": self.max_walk_levels,
            "order_ttl_ms": self.order_ttl_ms,
            "charge_maker_as_taker": self.charge_maker_as_taker,
        }


#: The default. Named so a result can state which model produced it.
PESSIMISTIC: CostModel = CostModel()

#: Costs switched off, for measuring cost sensitivity by difference. Never a
#: basis for promoting a strategy; the engine records which model was used.
FRICTIONLESS: CostModel = CostModel(
    fee_schedule=FeeSchedule.from_series("flat", 0.0000001),
    latency=LatencyModel(decision_to_arrival_ms=0, jitter_ms=0, cancel_ms=0),
    queue_ahead_bps=0,
    taker_concession_cc=0,
)


@dataclass(slots=True)
class CostReport:
    """Running total of what the simulated costs actually came to.

    Kept separately from the profit-and-loss figures so that "the strategy made
    money" and "the strategy made money *after* $X of fees and Y cc of average
    slippage" are two different, separately checkable statements.
    """

    fees_micros: int = 0
    taker_fills: int = 0
    maker_fills: int = 0
    filled_qc: int = 0
    #: Size-weighted slippage against the decision-time touch, in cc * qc.
    slippage_cc_qc: int = 0
    latency_samples: list[int] = field(default_factory=list)
    rejected_post_only: int = 0
    expired_orders: int = 0
    unfilled_orders: int = 0

    def record_fill(
        self, *, qty_qc: int, fee_micros: int, is_taker: bool, slippage_cc: int
    ) -> None:
        self.fees_micros += fee_micros
        self.filled_qc += qty_qc
        self.slippage_cc_qc += slippage_cc * qty_qc
        if is_taker:
            self.taker_fills += 1
        else:
            self.maker_fills += 1

    def record_latency(self, latency_ms: int) -> None:
        self.latency_samples.append(latency_ms)

    @property
    def fill_count(self) -> int:
        return self.taker_fills + self.maker_fills

    @property
    def avg_slippage_cc(self) -> float:
        """Size-weighted average slippage in centicents. Positive is a cost."""
        if self.filled_qc <= 0:
            return 0.0
        return self.slippage_cc_qc / self.filled_qc

    @property
    def maker_fill_pct(self) -> float:
        total = self.fill_count
        return 0.0 if total == 0 else self.maker_fills / total

    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fees_usd": str(micros_to_decimal(self.fees_micros)),
            "fees_micros": self.fees_micros,
            "fill_count": self.fill_count,
            "taker_fills": self.taker_fills,
            "maker_fills": self.maker_fills,
            "maker_fill_pct": round(self.maker_fill_pct, 4),
            "filled_qc": self.filled_qc,
            "avg_slippage_cc": round(self.avg_slippage_cc, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "rejected_post_only": self.rejected_post_only,
            "expired_orders": self.expired_orders,
            "unfilled_orders": self.unfilled_orders,
        }
