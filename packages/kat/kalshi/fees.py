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
from typing import Any, Final

from kat.common.fixedpoint import CC_PER_DOLLAR, MICROS_PER_DOLLAR, QC_PER_CONTRACT

__all__ = [
    "BASE_COEFFICIENT",
    "FALLBACK_MULTIPLIER",
    "FeeEstimate",
    "FeeSchedule",
    "PerpFeeSchedule",
    "estimate_fee_micros",
    "perp_fee_micros",
    "prorate_fee_micros",
    "realised_fee_micros",
    "round_trip_fee_micros",
]

#: Kalshi's published general trading fee coefficient -- the ``m`` in
#: ``m * C * P * (1 - P)``. The API does not carry it; the exchange publishes
#: only a per-series *scale* on it. So this constant is the only place the base
#: rate exists, and :meth:`FeeSchedule.from_exchange_series` is the only thing
#: that applies a scale to it.
BASE_COEFFICIENT: Final[Decimal] = Decimal("0.07")

#: The same number in its other job: what a schedule falls back to when the
#: series terms are unknown, which is also when the estimate is marked inexact.
FALLBACK_MULTIPLIER: Final[Decimal] = BASE_COEFFICIENT

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

        ``fee_multiplier`` here is the **coefficient** -- the whole ``m`` in
        ``m * C * P * (1 - P)`` -- which is what every caller in this codebase
        passes and what :func:`estimate_fee_micros` consumes.

        It is *not* the ``fee_multiplier`` field the exchange returns. That one
        is a scale on Kalshi's published base rate and reads ``1`` for a standard
        series. Use :meth:`from_exchange_series` for a payload, which converts;
        passing the exchange's value straight in here would price a $0.50
        contract at 25 cents a side instead of 1.75.

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

    @classmethod
    def from_exchange_series(
        cls,
        payload: dict[str, Any] | None,
        *,
        fee_type_override: str | None = None,
        fee_multiplier_override: float | None = None,
    ) -> FeeSchedule:
        """Build a schedule from Kalshi's own ``GET /series/{ticker}`` response.

        The exchange's ``fee_multiplier`` is a scale on its published base rate,
        not the rate itself. The base coefficient lives in the fee schedule PDF
        and never in the API, which is why :data:`BASE_COEFFICIENT` has to be
        supplied from this side. Every series sampled on production returns
        exactly ``1``, meaning "charged at the standard rate".

        Reading that ``1`` as the coefficient is the trap this method exists to
        close, and it is a quiet one. The model would become
        ``1.0 * C * P * (1 - P)``: 25 cents on a $0.50 contract rather than 1.75,
        fourteen times the truth, on a schedule marked *exact* so nothing
        downstream would pad it or doubt it. Every expected value in the platform
        would go negative at once and the strategies would stop trading, for a
        reason no log would name.
        """
        data = payload or {}
        raw_type = fee_type_override or data.get("fee_type") or "quadratic"
        raw_scale = (
            fee_multiplier_override
            if fee_multiplier_override is not None
            else data.get("fee_multiplier")
        )
        if raw_scale is None:
            return cls(fee_type=str(raw_type), multiplier=BASE_COEFFICIENT, exact=False)
        try:
            scale = Decimal(str(raw_scale))
        except (ArithmeticError, ValueError):
            return cls(fee_type=str(raw_type), multiplier=BASE_COEFFICIENT, exact=False)
        if scale <= 0:
            # A free market is not something the exchange offers, so a zero or
            # negative scale is a payload we do not understand rather than a
            # discount. Fall back, and stay inexact so the margin is applied.
            return cls(fee_type=str(raw_type), multiplier=BASE_COEFFICIENT, exact=False)
        return cls(fee_type=str(raw_type), multiplier=BASE_COEFFICIENT * scale, exact=True)

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
    apply_safety_margin: bool = True,
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

    if not schedule.exact and apply_safety_margin:
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


def realised_fee_micros(
    *,
    price_cc: int,
    qty_qc: int,
    schedule: FeeSchedule | None = None,
    is_taker: bool = True,
) -> int:
    """What the exchange would actually charge for a fill that happened.

    Distinct from :func:`estimate_fee_micros` in exactly one way, and the
    difference is the whole reason this exists. An estimate is used to decide
    whether to trade, so when the coefficient is unknown it is padded by
    :data:`INEXACT_SAFETY_MARGIN` -- deliberately too high, so an unknown fee
    schedule cannot make a marginal trade look profitable. That padding is a
    property of the decision, not of the market.

    Charging it is a different claim: that the exchange took 50% more than the
    published schedule says. It did not. Billing a padded estimate as a realised
    cost makes simulated results worse than the reality they exist to predict,
    which is the direction that quietly rejects strategies that would have
    worked.

    Only the simulated-fill path needs this. Live and demo take the fee from the
    exchange's own fill payload and never model it at all.
    """
    return estimate_fee_micros(
        price_cc=price_cc,
        qty_qc=qty_qc,
        schedule=schedule,
        is_taker=is_taker,
        apply_safety_margin=False,
    ).fee_micros


# ---------------------------------------------------------------------------
# Perpetuals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerpFeeSchedule:
    """Maker and taker rates for one perpetual, as fractions of notional.

    Perpetuals are priced nothing like event contracts and must not borrow their
    fee model. An event contract pays ``m * C * P * (1 - P)``, which peaks at
    175 centicents a side on a fifty-cent contract -- 350 basis points of the
    notional traded. A perpetual pays a flat rate on notional: at the time of
    writing every Kalshi perpetual charges 2bp to a maker and 4bp to a taker,
    which is roughly two orders of magnitude cheaper.

    Running perps through :class:`FeeSchedule` charges the quadratic fee on a
    linear instrument and overstates the cost about a hundredfold, which does
    not merely distort a backtest -- it rejects every strategy that would have
    worked. The gap mattered: the perpetuals domain has a REST client, three
    strategies, a risk engine and an execution engine, and no perp result had
    ever been costed with a perp fee.

    ``exact`` is False when the rates did not come from the exchange, and the
    same refusal-to-guess posture applies as for event contracts.
    """

    maker_rate: Decimal = Decimal("0.0004")
    taker_rate: Decimal = Decimal("0.0004")
    exact: bool = False

    @classmethod
    def from_fee_tiers(
        cls, payload: Any, ticker: str
    ) -> PerpFeeSchedule:
        """Build from ``GetMarginFeeTiers``, which quotes rates per market.

        A market missing from the response keeps the conservative default --
        the taker rate on both sides -- rather than inheriting another market's
        discount.
        """
        makers = (payload or {}).get("maker_fee_rates") or {}
        takers = (payload or {}).get("taker_fee_rates") or {}
        maker, taker = makers.get(ticker), takers.get(ticker)
        if maker is None or taker is None:
            return cls()
        try:
            return cls(
                maker_rate=Decimal(str(maker)),
                taker_rate=Decimal(str(taker)),
                exact=True,
            )
        except (ArithmeticError, TypeError, ValueError):
            return cls()


def perp_fee_micros(
    *,
    price_cc: int,
    qty_qc: int,
    schedule: PerpFeeSchedule | None = None,
    is_taker: bool = True,
) -> int:
    """Fee for one perpetual fill, in micros.

    ``qty_qc * price_cc`` is already micros of notional -- centi-contracts times
    ten-thousandths of a dollar -- so the rate applies to that product directly
    with no rescaling. Rounded up, because the exchange does not round a fee
    down in our favour.
    """
    if price_cc <= 0 or qty_qc <= 0:
        return 0
    schedule = schedule or PerpFeeSchedule()
    rate = schedule.taker_rate if is_taker else schedule.maker_rate
    notional_micros = Decimal(qty_qc) * Decimal(price_cc)
    return int((notional_micros * rate).to_integral_value(rounding=ROUND_CEILING))
