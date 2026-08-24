"""What an arbitrage is actually worth.

Most published prediction-market "arbitrage" is an artefact of quoting the
midpoint. A YES at $0.48 and a NO at $0.50 look like a two-cent free lunch until
you notice you must *cross the spread on both legs* and pay a fee on each, at
which point it is a two-cent loss.

This module exists so that never happens here. Every candidate is priced by
walking the real book for the size actually intended, charging the real fee on
every leg, and then asking what happens when the trade only half works.

Leg risk is the part everyone forgets
-------------------------------------
A two-leg arbitrage is not one trade, it is two trades that must both fill. If
leg one fills and leg two does not, you are left holding a naked directional
position that you entered *because you thought you were hedged* -- which is the
worst possible reason to hold it.

:func:`price_opportunity` therefore computes:

* the **expected value** if everything fills,
* the **execution probability**, from the depth available at each leg's price,
* the **residual exposure** if the hardest leg fails, and
* the **worst case**, being the loss on the filled legs when the unfilled one
  has to be entered at the far touch instead.

An opportunity is only surfaced when its expected value is positive *after* all
of that, not merely when its gross edge is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from kat.common.enums import OutcomeSide
from kat.common.fixedpoint import CC_PER_DOLLAR, micros_to_decimal
from kat.kalshi.fees import FeeSchedule, estimate_fee_micros
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["ArbLeg", "PricedOpportunity", "price_opportunity"]

#: A YES and a NO contract on the same market together settle at exactly $1.00.
PAIR_VALUE_CC = CC_PER_DOLLAR

#: Fraction of resting size at a price we assume we can actually take. The book
#: shows what is displayed, not what will still be there when our order lands,
#: and an arbitrage that needs the whole displayed size is one that will not fill.
DEPTH_CONFIDENCE = Decimal("0.5")


@dataclass(slots=True)
class ArbLeg:
    """One leg of a candidate arbitrage."""

    ticker: str
    side: OutcomeSide
    #: Price we would have to pay, in centicents, having walked the book.
    price_cc: int
    qty_qc: int
    #: Size resting at or better than ``price_cc`` on the side we must take.
    available_qc: int
    fee_schedule: FeeSchedule | None = None
    #: The far touch, used to price what re-entering this leg would cost if the
    #: other legs filled and this one did not.
    fallback_price_cc: int | None = None

    @property
    def cost_micros(self) -> int:
        return self.price_cc * self.qty_qc

    @property
    def fee_micros(self) -> int:
        return estimate_fee_micros(
            price_cc=self.price_cc,
            qty_qc=self.qty_qc,
            schedule=self.fee_schedule,
            is_taker=True,
        ).fee_micros

    @property
    def fee_is_exact(self) -> bool:
        return (self.fee_schedule or FeeSchedule()).exact


@dataclass(slots=True)
class PricedOpportunity:
    """A candidate arbitrage with its full economics attached."""

    kind: str
    legs: list[ArbLeg]
    #: What the legs settle to in total, regardless of outcome. For a complete
    #: YES+NO pair this is $1.00 per contract.
    settlement_micros: int
    gross_edge_micros: int
    fees_micros: int
    slippage_micros: int
    net_edge_micros: int
    execution_probability: Decimal
    residual_exposure_micros: int
    worst_case_micros: int
    expected_value_micros: int
    qty_qc: int
    fees_are_exact: bool = True
    reason: str = ""
    detail: dict[str, object] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Whether this is worth doing, all costs and leg risk considered."""
        return self.expected_value_micros > 0 and self.qty_qc > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "qty": str(Decimal(self.qty_qc) / 100),
            "legs": [
                {
                    "ticker": leg.ticker,
                    "side": leg.side.value,
                    "price": str(Decimal(leg.price_cc) / CC_PER_DOLLAR),
                    "qty": str(Decimal(leg.qty_qc) / 100),
                }
                for leg in self.legs
            ],
            "gross_edge_usd": str(micros_to_decimal(self.gross_edge_micros)),
            "fees_usd": str(micros_to_decimal(self.fees_micros)),
            "slippage_usd": str(micros_to_decimal(self.slippage_micros)),
            "net_edge_usd": str(micros_to_decimal(self.net_edge_micros)),
            "execution_probability": f"{self.execution_probability:.4f}",
            "residual_exposure_usd": str(micros_to_decimal(self.residual_exposure_micros)),
            "worst_case_usd": str(micros_to_decimal(self.worst_case_micros)),
            "expected_value_usd": str(micros_to_decimal(self.expected_value_micros)),
            "actionable": self.is_actionable,
            "fees_are_exact": self.fees_are_exact,
            "reason": self.reason,
            **self.detail,
        }


def price_opportunity(
    *,
    kind: str,
    legs: list[ArbLeg],
    settlement_per_contract_micros: int,
    qty_qc: int,
) -> PricedOpportunity:
    """Compute the honest economics of a multi-leg arbitrage.

    ``settlement_per_contract_micros`` is what one complete set of legs pays out
    at resolution, whatever the outcome. For a YES+NO pair on one market that is
    $1.00; for a complete set of mutually exclusive outcomes it is also $1.00.

    The result is always returned, even when it is not worth doing. Recording
    declined opportunities is how the cost model gets validated: if the platform
    never declines anything, the costs are being understated somewhere.
    """
    if qty_qc <= 0 or not legs:
        return _nothing(kind, legs, "no size")

    total_cost = sum(leg.cost_micros for leg in legs)
    total_fees = sum(leg.fee_micros for leg in legs)
    settlement = settlement_per_contract_micros * qty_qc // 100

    gross_edge = settlement - total_cost
    net_edge = gross_edge - total_fees

    # --- execution probability from available depth ------------------------
    # Each leg must fill. Treating them as independent is a simplification, and
    # a conservative one: correlated failure (the whole book pulling at once) is
    # more likely than independence implies, so the true probability is lower
    # than this estimate, not higher.
    probability = Decimal(1)
    for leg in legs:
        usable = Decimal(leg.available_qc) * DEPTH_CONFIDENCE
        if usable <= 0:
            probability = Decimal(0)
            break
        leg_probability = min(Decimal(1), usable / Decimal(leg.qty_qc))
        probability *= leg_probability

    # --- what failure costs -------------------------------------------------
    # The hardest leg is the one most likely to fail, so that is the one modelled
    # as failing. If it does, we hold the other legs unhedged.
    hardest = min(legs, key=lambda leg: leg.available_qc)
    residual_exposure = sum(leg.cost_micros for leg in legs if leg is not hardest)

    # Worst case: we complete the position at the far touch instead. Where no
    # fallback price is known, assume the worst admissible price for that side,
    # because an unknown recovery cost is not a free one.
    fallback = hardest.fallback_price_cc
    if fallback is None:
        fallback = PAIR_VALUE_CC
    recovery_extra = max(0, (fallback - hardest.price_cc)) * qty_qc
    worst_case = net_edge - recovery_extra

    # --- expected value -----------------------------------------------------
    # p * (net edge) + (1 - p) * (worst case). The failure branch is charged in
    # full rather than assumed recoverable at no cost.
    ev = (
        probability * Decimal(net_edge) + (Decimal(1) - probability) * Decimal(worst_case)
    ).to_integral_value()

    fees_exact = all(leg.fee_is_exact for leg in legs)
    if not fees_exact:
        # An inexact fee estimate already carries a safety margin inside
        # estimate_fee_micros; flag it so the scanner can require a wider edge.
        pass

    reason = ""
    if gross_edge <= 0:
        reason = "no gross edge"
    elif net_edge <= 0:
        reason = "fees consume the gross edge"
    elif ev <= 0:
        reason = "leg risk consumes the net edge"
    elif probability < Decimal("0.5"):
        reason = "insufficient depth to expect both legs to fill"

    return PricedOpportunity(
        kind=kind,
        legs=legs,
        settlement_micros=settlement,
        gross_edge_micros=gross_edge,
        fees_micros=total_fees,
        slippage_micros=0,  # slippage is already inside each leg's walked price
        net_edge_micros=net_edge,
        execution_probability=probability,
        residual_exposure_micros=residual_exposure,
        worst_case_micros=worst_case,
        expected_value_micros=int(ev),
        qty_qc=qty_qc,
        fees_are_exact=fees_exact,
        reason=reason,
        detail={"total_cost_usd": str(micros_to_decimal(total_cost))},
    )


def _nothing(kind: str, legs: list[ArbLeg], reason: str) -> PricedOpportunity:
    return PricedOpportunity(
        kind=kind,
        legs=legs,
        settlement_micros=0,
        gross_edge_micros=0,
        fees_micros=0,
        slippage_micros=0,
        net_edge_micros=0,
        execution_probability=Decimal(0),
        residual_exposure_micros=0,
        worst_case_micros=0,
        expected_value_micros=0,
        qty_qc=0,
        reason=reason,
    )


def walk_for_size(
    book: OrderBook, qty_qc: int, *, is_buy: bool
) -> tuple[int, int] | None:
    """Average price and available size for taking ``qty_qc`` from ``book``.

    Returns ``(vwap_cc, available_qc)``, or ``None`` when the side is empty.
    The VWAP is the price that matters: an arbitrage sized at ten contracts must
    be priced at the average of the ten, not at the touch.
    """
    side_depth = book.total_depth_qc(is_bid=not is_buy)
    if side_depth <= 0:
        return None
    vwap = book.vwap_cc(min(qty_qc, side_depth), is_buy=is_buy)
    if vwap is None:
        return None
    return vwap, side_depth
