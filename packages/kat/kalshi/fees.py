"""Fee modelling.

Fees are not a rounding detail on a sub-$250 account trading contracts worth
cents: a 7% quadratic fee on a $0.50 contract is $0.0175 per contract, which is
larger than most of the edges a strategy will find. Getting this wrong makes
every expected-value calculation wrong in the optimistic direction, so this
module is deliberately conservative and explicit about what it does not know.

What the API tells us
---------------------
``GET /series/{ticker}`` returns ``fee_type`` and ``fee_multiplier`` for each
series, and events can override both. The documented fee types are:

``quadratic``
    The General Trading Fees table: fee is proportional to ``C * P * (1 - P)``,
    which peaks at a $0.50 price and falls to nearly nothing at the extremes.
``quadratic_with_maker_fees``
    Same taker formula, plus a maker fee at a 0.25 multiplier.
``quadratic_with_combo_maker_fees``
    Same, with a 0.5 maker multiplier.
``flat``
    The Specific Trading Fees table: a flat rate per contract.

What the API does *not* tell us
-------------------------------
The base coefficient itself lives in Kalshi's published fee schedule PDF, not in
the API. The worked examples in Kalshi's fee-rounding documentation are drawn
from markets with different multipliers and cannot be used to derive a single
constant -- one of them is consistent with 0.07 and another is not.

This module therefore **never assumes a coefficient it was not given**:

* When the series ``fee_multiplier`` is known, it is used and the estimate is
  marked exact.
* When it is not known, :data:`FALLBACK_MULTIPLIER` is used, the estimate is
  marked **inexact**, and a safety margin is added. The risk engine refuses to
  trade on an inexact fee estimate whose margin would consume the edge.

Realised fees always come from the fill payload (``fee_cost``/``taker_fees``),
never from this model. The model exists to decide whether to trade; the exchange
decides what was actually charged.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Final

from kat.common.fixedpoint import CC_PER_DOLLAR, MICROS_PER_DOLLAR, QC_PER_CONTRACT

__all__ = [
    "FALLBACK_MULTIPLIER",
    "FeeEstimate",
    "FeeSchedule",
    "estimate_fee_micros",
    "prorate_fee_micros",
    "round_trip_fee_micros",
]

#: Kalshi's published general trading fee coefficient. Used only when the series
#: multiplier is unavailable, and always flagged as an inexact estimate.
FALLBACK_MULTIPLIER: Final[Decimal] = Decimal("0.07")

#: Extra margin applied to an inexact estimate, so an unknown fee schedule
#: cannot make a marginal trade look profitable.
INEXACT_SAFETY_MARGIN: Final[Decimal] = Decimal("1.50")

#: Maker multipliers documented for the two maker-fee variants.
_MAKER_MULTIPLIERS: Final[dict[str, Decimal]] = {
    "quadratic_with_maker_fees": Decimal("0.25"),
    "quadratic_with_combo_maker_fees": Decimal("0.50"),
}

#: Fees are charged in whole centicents, rounded up. See Kalshi's fee-rounding
#: documentation: the trade fee is ceiled to $0.0001 before the balance-rounding
#: step is applied.
_CENTICENT: Final[Decimal] = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """The fee terms in force for one market.

    Built from the series (and any event override) and cached. ``exact`` is
    False when we had to fall back to a default multiplier.
    """

    fee_type: str = "quadratic"
    multiplier: Decimal = FALLBACK_MULTIPLIER
    exact: bool = False

    @classmethod
    def from_series(
        cls,
        fee_type: str | None,
        fee_multiplier: float | None,
        *,
        fee_type_override: str | None = None,
        fee_multiplier_override: float | None = None,
    ) -> FeeSchedule:
        """Build a schedule, preferring an event-level override over the series.

        A null override means "cleared", not "zero", so a null falls back to the
        series value rather than to a free market.
        """
        resolved_type = fee_type_override or fee_type or "quadratic"
        raw_multiplier = (
            fee_multiplier_override if fee_multiplier_override is not None else fee_multiplier
        )
        if raw_multiplier is None or raw_multiplier <= 0:
            return cls(fee_type=resolved_type, multiplier=FALLBACK_MULTIPLIER, exact=False)
        return cls(
            fee_type=resolved_type,
            # via str() so 0.07 means the literal 0.07, not its binary expansion
            multiplier=Decimal(str(raw_multiplier)),
            exact=True,
        )

    @property
    def maker_multiplier(self) -> Decimal | None:
        """Maker-side multiplier, or ``None`` when the series has no maker fee."""
        return _MAKER_MULTIPLIERS.get(self.fee_type)


@dataclass(frozen=True, slots=True)
class FeeEstimate:
    """An estimated fee, with the honesty of the estimate attached."""

    fee_micros: int
    exact: bool
    fee_type: str
    is_taker: bool

    @property
    def usd(self) -> Decimal:
        return Decimal(self.fee_micros) / Decimal(MICROS_PER_DOLLAR)


def estimate_fee_micros(
    *,
    price_cc: int,
    qty_qc: int,
    schedule: FeeSchedule | None = None,
    is_taker: bool = True,
) -> FeeEstimate:
    """Estimate the trading fee for ``qty_qc`` contracts at ``price_cc``.

    The quadratic model is ``multiplier * contracts * price * (1 - price)``,
    ceiled to the nearest centicent, matching the documented rounding.

    ``is_taker=False`` is only honoured for series that actually have a maker
    fee; for a plain ``quadratic`` series a maker fill is charged the same as a
    taker fill, so passing ``is_taker=False`` does not reduce the estimate. When
    the caller does not know whether a fill will be maker or taker, leaving the
    default of ``True`` gives the conservative answer.
    """
    schedule = schedule or FeeSchedule()

    if qty_qc <= 0 or price_cc <= 0:
        return FeeEstimate(0, schedule.exact, schedule.fee_type, is_taker)

    contracts = Decimal(qty_qc) / Decimal(QC_PER_CONTRACT)
    price = Decimal(price_cc) / Decimal(CC_PER_DOLLAR)

    if schedule.fee_type == "flat":
        # Specific Trading Fees: a flat rate per contract, independent of price.
        raw = schedule.multiplier * contracts
    else:
        multiplier = schedule.multiplier
        if not is_taker:
            maker = schedule.maker_multiplier
            if maker is not None:
                multiplier = multiplier * maker
        raw = multiplier * contracts * price * (Decimal(1) - price)

    if not schedule.exact:
        # We are guessing at the coefficient; guess high.
        raw = raw * INEXACT_SAFETY_MARGIN

    fee_dollars = raw.quantize(_CENTICENT, rounding=ROUND_CEILING)
    fee_micros = int(fee_dollars * Decimal(MICROS_PER_DOLLAR))
    return FeeEstimate(
        fee_micros=fee_micros,
        exact=schedule.exact,
        fee_type=schedule.fee_type,
        is_taker=is_taker,
    )


def round_trip_fee_micros(
    *,
    entry_price_cc: int,
    exit_price_cc: int,
    qty_qc: int,
    schedule: FeeSchedule | None = None,
) -> int:
    """Total fee for entering and exiting a position.

    A strategy that only counts the entry fee will systematically overstate its
    edge by roughly a factor of two, so expected-value calculations use this.
    """
    entry = estimate_fee_micros(
        price_cc=entry_price_cc, qty_qc=qty_qc, schedule=schedule, is_taker=True
    )
    exit_ = estimate_fee_micros(
        price_cc=exit_price_cc, qty_qc=qty_qc, schedule=schedule, is_taker=True
    )
    return entry.fee_micros + exit_.fee_micros


def prorate_fee_micros(fee_micros: int, *, from_qty_qc: int, to_qty_qc: int) -> int:
    """Rescale a fee estimate from one quantity to another.

    A risk engine resizes an order after the fee has already been estimated for
    the quantity the strategy asked for, and it holds a number rather than the
    schedule that produced it, so it cannot simply recompute. It can rescale,
    because every fee model in this module is linear in quantity: the quadratic
    forms are ``multiplier * contracts * price * (1 - price)`` and the flat form
    is ``multiplier * contracts``. Only the price terms are non-linear, and a
    resize does not change the price.

    That linearity is the whole justification for this function. A fee model
    with a size-dependent rate -- a tiered or capped schedule -- would break it,
    which is why the assumption is written down here rather than left implicit
    at the call sites.

    Rounds up, so rescaling never invents a cheaper trade than the estimate did.
    """
    if to_qty_qc <= 0 or fee_micros <= 0:
        return 0
    if from_qty_qc <= 0 or to_qty_qc >= from_qty_qc:
        return fee_micros
    return -(-fee_micros * to_qty_qc // from_qty_qc)
