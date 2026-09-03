"""Polymarket US fee modelling, where the maker is paid rather than charged.

This venue exists in the platform for one reason, and it is a single number.
Both exchanges price event contracts on the same quadratic form::

    fee = theta * contracts * price * (1 - price)

and differ only in ``theta``:

======================  ==========  ===============
                        Kalshi      Polymarket US
======================  ==========  ===============
taker theta             0.07        0.06
maker theta             0.07        **-0.0125**
minimum tick            $0.01       $0.001
minimum order           1 contract  1 contract
======================  ==========  ===============

A negative coefficient is not a discount, it is a reversal. On a market quoted
15.4/16.0 -- a real book, sampled live -- a Kalshi maker pays 92.6 centicents a
side and turns a 60-centicent spread into a 125-centicent loss. A Polymarket
maker is *paid* 16.5 a side and turns the same spread into a 93-centicent gain.
That is a 218-centicent swing on a contract worth 1,570, and it is the reason
market making measured -0.49% on Kalshi is worth measuring again here.

The tick matters nearly as much. Kalshi's minimum increment is a whole cent,
which on a three-cent contract is a third of its notional; that single fact
priced out every cheap market on the board. A tenth of a cent is a tenth of that
obstacle.

Why this is not :mod:`kat.kalshi.fees` with a different constant
----------------------------------------------------------------
Because a negative fee runs the wrong way through arithmetic written for costs,
in three specific places, and each one fails silently in the profitable
direction:

*Rounding.* Ceiling happens to be right for both signs -- it makes a cost
larger and a rebate smaller -- but only once that is checked rather than
inherited. The first draft of this module branched on the sign and rounded
rebates the wrong way.

*The safety margin.* Multiplying an uncertain cost by 1.5 makes it worse, which
is the point. Multiplying an uncertain *rebate* by 1.5 makes it better, which is
the exact inverse of a safety margin. Here the margin divides a rebate instead.

*The sign convention downstream.* ``net = gross - fees`` is correct for both, and
that is precisely why this must be signed rather than an absolute magnitude with
a flag: a caller that takes ``abs()`` anywhere turns income into cost without
raising anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Any, Final

from kat.common.fixedpoint import CC_PER_DOLLAR, MICROS_PER_DOLLAR, QC_PER_CONTRACT

__all__ = [
    "MAKER_THETA",
    "TAKER_THETA",
    "PolymarketFeeSchedule",
    "polymarket_fee_micros",
    "polymarket_round_trip_micros",
]

#: Published taker coefficient. Markets carry it per-market as ``feeCoefficient``
#: and every market sampled so far reports exactly this, but it is read from the
#: market rather than assumed -- the same posture the Kalshi model takes.
TAKER_THETA: Final[Decimal] = Decimal("0.06")

#: Published maker coefficient. Negative: the maker is paid. There is no volume
#: gate on it -- the tiered rebates in the fee schedule are an *additional*
#: discount on taker volume above $250k a month, not a condition on this.
MAKER_THETA: Final[Decimal] = Decimal("-0.0125")

#: Applied when a coefficient did not come from the exchange. A cost is
#: multiplied by it and a rebate divided, so an unknown schedule can never make
#: a marginal trade look better than it is.
INEXACT_SAFETY_MARGIN: Final[Decimal] = Decimal("1.50")

_CENTICENT: Final[Decimal] = Decimal("0.0001")


@dataclass(frozen=True)
class PolymarketFeeSchedule:
    """The fee terms in force for one Polymarket market.

    ``exact`` is False when the taker coefficient did not come from the market
    payload. The maker coefficient is not published per-market, so it is always
    the documented constant.
    """

    taker_theta: Decimal = TAKER_THETA
    maker_theta: Decimal = MAKER_THETA
    exact: bool = False

    @classmethod
    def from_market(cls, payload: Any) -> PolymarketFeeSchedule:
        """Build from a market's ``feeCoefficient``.

        A market that does not publish one keeps the documented default and is
        marked inexact, rather than being assumed to match its neighbours.
        """
        raw = (payload or {}).get("feeCoefficient")
        if raw in (None, ""):
            return cls()
        try:
            theta = Decimal(str(raw))
        except (ArithmeticError, TypeError, ValueError):
            return cls()
        if theta < 0:
            # A negative *taker* coefficient would mean the exchange pays both
            # sides of every trade, which is not a fee schedule, it is a
            # misreading. Refuse it rather than bank on it.
            return cls()
        return cls(taker_theta=theta, exact=True)

    @property
    def pays_makers(self) -> bool:
        return self.maker_theta < 0


def polymarket_fee_micros(
    *,
    price_cc: int,
    qty_qc: int,
    schedule: PolymarketFeeSchedule | None = None,
    is_taker: bool = True,
    apply_safety_margin: bool = True,
) -> int:
    """Signed fee for one fill, in micros. Negative means we are paid.

    The sign is the whole point and is deliberately not hidden behind a
    magnitude-plus-flag: ``net = gross - fee`` then reads correctly for a cost
    and for a rebate without the caller knowing which it has.
    """
    schedule = schedule or PolymarketFeeSchedule()
    if qty_qc <= 0 or price_cc <= 0 or price_cc >= CC_PER_DOLLAR:
        return 0

    contracts = Decimal(qty_qc) / Decimal(QC_PER_CONTRACT)
    price = Decimal(price_cc) / Decimal(CC_PER_DOLLAR)
    theta = schedule.taker_theta if is_taker else schedule.maker_theta
    raw = theta * contracts * price * (Decimal(1) - price)

    if not schedule.exact and apply_safety_margin:
        # Guess high on a cost, low on a rebate. Multiplying both would inflate
        # the rebate, which is a safety margin pointed the wrong way.
        raw = raw * INEXACT_SAFETY_MARGIN if raw > 0 else raw / INEXACT_SAFETY_MARGIN

    # One rule covers both signs. Ceiling rounds toward positive infinity, so
    # it makes a cost larger and a rebate smaller -- against us either way.
    # Branching on the sign here would be the mistake, not the fix.
    fee_dollars = raw.quantize(_CENTICENT, rounding=ROUND_CEILING)
    return int(fee_dollars * Decimal(MICROS_PER_DOLLAR))


def polymarket_round_trip_micros(
    *,
    entry_price_cc: int,
    exit_price_cc: int,
    qty_qc: int,
    schedule: PolymarketFeeSchedule | None = None,
    entry_is_taker: bool = True,
    exit_is_taker: bool = True,
) -> int:
    """Signed cost of both legs. Negative when the round trip is paid for.

    Both legs are priced at their own price, because the quadratic is not
    symmetric about the two and a position that moves is charged differently
    getting out than getting in.
    """
    return polymarket_fee_micros(
        price_cc=entry_price_cc, qty_qc=qty_qc,
        schedule=schedule, is_taker=entry_is_taker,
    ) + polymarket_fee_micros(
        price_cc=exit_price_cc, qty_qc=qty_qc,
        schedule=schedule, is_taker=exit_is_taker,
    )
