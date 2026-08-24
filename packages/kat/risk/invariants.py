"""THE HARD ARCHITECTURAL INVARIANT: perpetual leverage may never exceed 1.00x.

Read this before changing anything in this file.

What "1.00x" means here
-----------------------
Effective leverage is defined as::

    effective_leverage = total_absolute_position_notional / account_equity

Holding notional equal to your equity is 1.00x -- fully collateralised, nothing
borrowed. That is the maximum this platform will ever reach.

This is deliberately **not** the same metric Kalshi reports. Kalshi's
``GET /margin/risk`` returns ``account_leverage = total_position_notional /
total_maintenance_margin``, which is a margin-utilisation ratio: with a 10%
maintenance requirement, a completely unlevered position reports an
``account_leverage`` of 10. Using Kalshi's number as the invariant would permit
roughly 10x real leverage. We therefore compute our own number from equity, and
use the exchange's figures only as a corroborating second opinion (see
``kat.risk.perp_risk``).

How the limit is made unbypassable
----------------------------------
There is no leverage parameter in this module. The check is the integer
comparison::

    projected_exposure_micros <= equity_micros

There is no ratio to configure, no multiplier to scale, no environment variable
to read, and no code path that accepts a caller-supplied maximum. To raise the
limit you would have to edit this comparison itself -- which is the point.
``MAX_PERP_LEVERAGE`` exists only so the dashboard has something to display; the
enforcement path never reads it.

Because the comparison is over exact integers in micro-dollars
(see :mod:`kat.common.fixedpoint`), 1.0000x passes and 1.0001x fails by exactly
one part in ten thousand, with no floating-point ambiguity anywhere.

Tamper detection
----------------
``assert_invariant_intact()`` runs a set of behavioural canaries -- known inputs
with known verdicts -- on every enforcement call. If any layer of the system
(a strategy, a model, an LLM tool call, a plugin, a monkeypatch in a test that
escaped into production) has replaced or weakened the evaluation logic, the
canaries fail and the platform refuses to trade rather than trading unsafely.

De-risking is not a bypass
--------------------------
An order that *strictly reduces* exposure is permitted even when current
leverage already exceeds 1.00x. This is not an exemption: mark-price moves can
push leverage above the line with no order of ours involved, and forbidding the
only orders that fix it would be the unsafe choice. An order claiming to reduce
exposure that would in fact increase it is rejected like any other.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from kat.common.errors import LeverageViolation
from kat.common.fixedpoint import MICROS_PER_DOLLAR, micros_to_decimal

__all__ = [
    "MAX_PERP_LEVERAGE",
    "MAX_PERP_LEVERAGE_BASIS_POINTS",
    "LeverageAssessment",
    "assert_invariant_intact",
    "enforce_perp_leverage",
    "evaluate_perp_leverage",
    "position_exposure_micros",
]

# ---------------------------------------------------------------------------
# Display constants. NOT used by the enforcement path -- see module docstring.
# ---------------------------------------------------------------------------

#: The absolute maximum effective leverage for Kalshi perpetuals. Architectural,
#: not configurable. Present so the dashboard and audit records can state it.
MAX_PERP_LEVERAGE: Final[Decimal] = Decimal("1.00")

#: Same limit in basis points, for integer-only consumers.
MAX_PERP_LEVERAGE_BASIS_POINTS: Final[int] = 10_000

# Reason codes emitted on rejection. Stable strings: the dashboard and the audit
# log both key off them.
_R_OK: Final[str] = "WITHIN_LIMIT"
_R_REDUCTION: Final[str] = "PERMITTED_EXPOSURE_REDUCTION"
_R_EXCEEDS: Final[str] = "LEVERAGE_WOULD_EXCEED_1X"
_R_NO_EQUITY: Final[str] = "EQUITY_UNKNOWN_OR_NON_POSITIVE"
_R_MALFORMED: Final[str] = "MALFORMED_RISK_INPUT"
_R_NEGATIVE_EXPOSURE: Final[str] = "NEGATIVE_ABSOLUTE_EXPOSURE"
_R_TAMPERED: Final[str] = "INVARIANT_SELF_CHECK_FAILED"


@dataclass(frozen=True, slots=True)
class LeverageAssessment:
    """The full arithmetic behind one leverage verdict.

    Every field is recorded on the decision record and rendered on the
    Perpetuals dashboard tab, so that any rejection can be explained without
    re-running anything.
    """

    equity_micros: int
    current_exposure_micros: int
    pending_exposure_micros: int
    requested_delta_micros: int
    projected_exposure_micros: int
    permitted: bool
    reason_code: str
    reason: str
    is_reduction: bool = False

    # -- derived, presentation-only -----------------------------------------

    @property
    def current_leverage(self) -> Decimal | None:
        """Leverage from filled positions alone. ``None`` when equity is unusable."""
        return _ratio(self.current_exposure_micros, self.equity_micros)

    @property
    def effective_leverage(self) -> Decimal | None:
        """Leverage including exposure locked by working orders."""
        return _ratio(
            self.current_exposure_micros + self.pending_exposure_micros, self.equity_micros
        )

    @property
    def projected_leverage(self) -> Decimal | None:
        """Leverage that would result if the requested order filled completely."""
        return _ratio(self.projected_exposure_micros, self.equity_micros)

    @property
    def max_leverage(self) -> Decimal:
        return MAX_PERP_LEVERAGE

    @property
    def headroom_micros(self) -> int:
        """Additional notional that could still be taken on, never below zero."""
        used = self.current_exposure_micros + self.pending_exposure_micros
        return max(0, self.equity_micros - used)

    def to_dict(self) -> dict[str, object]:
        """Flat, JSON-safe form for audit rows and the dashboard API."""

        def _s(value: Decimal | None) -> str | None:
            # Six decimals, not four: a rejection at 1.000001x must not render
            # as "1.0000" and look identical to the 1.00x case it was refused for.
            return None if value is None else f"{value:.6f}"

        return {
            "equity_usd": str(micros_to_decimal(self.equity_micros)),
            "current_exposure_usd": str(micros_to_decimal(self.current_exposure_micros)),
            "pending_exposure_usd": str(micros_to_decimal(self.pending_exposure_micros)),
            "requested_delta_usd": str(micros_to_decimal(self.requested_delta_micros)),
            "projected_exposure_usd": str(micros_to_decimal(self.projected_exposure_micros)),
            "current_leverage": _s(self.current_leverage),
            "effective_leverage": _s(self.effective_leverage),
            "projected_leverage": _s(self.projected_leverage),
            "max_leverage": f"{MAX_PERP_LEVERAGE:.2f}",
            "headroom_usd": str(micros_to_decimal(self.headroom_micros)),
            "permitted": self.permitted,
            "is_reduction": self.is_reduction,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


def _ratio(exposure_micros: int, equity_micros: int) -> Decimal | None:
    if equity_micros <= 0:
        return None
    return (Decimal(exposure_micros) / Decimal(equity_micros)).quantize(Decimal("0.000001"))


def _is_plain_int(value: object) -> bool:
    """Accept ``int`` only -- and not ``bool``, which is an ``int`` subclass.

    Rejecting ``float`` here is what stops a malformed or float-derived value
    from entering the comparison. Callers convert through
    :mod:`kat.common.fixedpoint`, which is exact by construction.
    """
    return type(value) is int  # noqa: E721 - subclasses (bool) must be rejected


def position_exposure_micros(qty_qc: int, mark_price_cc: int) -> int:
    """Absolute notional of a signed perpetual position, in micro-dollars.

    Longs and shorts both consume exposure: a short position of 10 contracts at
    a $0.60 mark carries the same $6.00 of notional risk as the equivalent long.
    Netting long against short across *different* markets is never done here.
    """
    if not _is_plain_int(qty_qc) or not _is_plain_int(mark_price_cc):
        raise LeverageViolation(
            "position exposure requires exact integer inputs",
            detail={"reason_code": _R_MALFORMED, "qty_qc": repr(qty_qc), "mark": repr(mark_price_cc)},
        )
    if mark_price_cc < 0:
        raise LeverageViolation(
            "negative mark price cannot produce a valid exposure",
            detail={"reason_code": _R_MALFORMED, "mark": mark_price_cc},
        )
    return abs(qty_qc) * mark_price_cc


def evaluate_perp_leverage(
    *,
    equity_micros: int,
    current_exposure_micros: int,
    pending_exposure_micros: int,
    requested_delta_micros: int,
) -> LeverageAssessment:
    """Decide whether a perpetual order may proceed. Pure function, no I/O.

    Parameters
    ----------
    equity_micros:
        Account equity in micro-dollars, computed conservatively by
        :mod:`kat.risk.perp_risk` (rounded *down*). Must be strictly positive;
        an unknown, zero or negative equity means we cannot prove safety and
        every exposure-increasing order is refused.
    current_exposure_micros:
        Sum over markets of ``|position| * mark_price``. Non-negative.
    pending_exposure_micros:
        Additional notional that working orders could add if they all filled.
        Counted in full, because an order resting on the book is an order that
        can fill. Non-negative.
    requested_delta_micros:
        The *change* in total absolute exposure this order would cause. Positive
        for orders that open or extend a position, negative for orders that
        reduce or close one. The caller computes it per market from signed
        positions so that a sell against a long correctly reduces exposure.

    Returns
    -------
    LeverageAssessment
        ``permitted`` is the verdict; every input and derived figure is retained
        for the audit trail.
    """
    # --- input validation: malformed input is never "probably fine" ---------
    for name, value in (
        ("equity_micros", equity_micros),
        ("current_exposure_micros", current_exposure_micros),
        ("pending_exposure_micros", pending_exposure_micros),
        ("requested_delta_micros", requested_delta_micros),
    ):
        if not _is_plain_int(value):
            return LeverageAssessment(
                equity_micros=0,
                current_exposure_micros=0,
                pending_exposure_micros=0,
                requested_delta_micros=0,
                projected_exposure_micros=0,
                permitted=False,
                reason_code=_R_MALFORMED,
                reason=(
                    f"{name} must be an exact integer number of micro-dollars, "
                    f"got {type(value).__name__} {value!r}"
                ),
            )

    if current_exposure_micros < 0 or pending_exposure_micros < 0:
        return LeverageAssessment(
            equity_micros=equity_micros,
            current_exposure_micros=current_exposure_micros,
            pending_exposure_micros=pending_exposure_micros,
            requested_delta_micros=requested_delta_micros,
            projected_exposure_micros=0,
            permitted=False,
            reason_code=_R_NEGATIVE_EXPOSURE,
            reason="absolute exposure cannot be negative; refusing to evaluate",
        )

    used = current_exposure_micros + pending_exposure_micros
    # Exposure can never fall below zero even if a caller over-states a reduction.
    projected = used + requested_delta_micros
    if projected < 0:
        projected = 0

    is_reduction = requested_delta_micros < 0

    # --- equity must be known and positive ---------------------------------
    if equity_micros <= 0:
        # A reduction is still allowed: flattening into an unknown-equity state
        # lowers risk and is how the platform recovers.
        if is_reduction:
            return LeverageAssessment(
                equity_micros=equity_micros,
                current_exposure_micros=current_exposure_micros,
                pending_exposure_micros=pending_exposure_micros,
                requested_delta_micros=requested_delta_micros,
                projected_exposure_micros=projected,
                permitted=True,
                reason_code=_R_REDUCTION,
                reason="exposure-reducing order permitted while equity is unavailable",
                is_reduction=True,
            )
        return LeverageAssessment(
            equity_micros=equity_micros,
            current_exposure_micros=current_exposure_micros,
            pending_exposure_micros=pending_exposure_micros,
            requested_delta_micros=requested_delta_micros,
            projected_exposure_micros=projected,
            permitted=False,
            reason_code=_R_NO_EQUITY,
            reason=(
                "account equity is unknown or non-positive; no exposure-increasing "
                "perpetual order can be proven safe"
            ),
        )

    # --- THE INVARIANT ------------------------------------------------------
    # Exact integer comparison. Equality is 1.0000x and is allowed; one
    # micro-dollar more is not. There is no tolerance and no configurable ratio.
    if projected <= equity_micros:
        return LeverageAssessment(
            equity_micros=equity_micros,
            current_exposure_micros=current_exposure_micros,
            pending_exposure_micros=pending_exposure_micros,
            requested_delta_micros=requested_delta_micros,
            projected_exposure_micros=projected,
            permitted=True,
            reason_code=_R_OK,
            reason="projected exposure is within 1.00x of account equity",
            is_reduction=is_reduction,
        )

    # Over the line. Only a *strict* reduction of exposure may still proceed.
    if is_reduction and projected < used:
        return LeverageAssessment(
            equity_micros=equity_micros,
            current_exposure_micros=current_exposure_micros,
            pending_exposure_micros=pending_exposure_micros,
            requested_delta_micros=requested_delta_micros,
            projected_exposure_micros=projected,
            permitted=True,
            reason_code=_R_REDUCTION,
            reason=(
                "leverage already exceeds 1.00x; order permitted only because it "
                "strictly reduces exposure"
            ),
            is_reduction=True,
        )

    return LeverageAssessment(
        equity_micros=equity_micros,
        current_exposure_micros=current_exposure_micros,
        pending_exposure_micros=pending_exposure_micros,
        requested_delta_micros=requested_delta_micros,
        projected_exposure_micros=projected,
        permitted=False,
        reason_code=_R_EXCEEDS,
        reason=(
            f"projected exposure {projected / MICROS_PER_DOLLAR:.6f} USD exceeds account "
            f"equity {equity_micros / MICROS_PER_DOLLAR:.6f} USD; the hard maximum is "
            f"{MAX_PERP_LEVERAGE:.2f}x"
        ),
    )


# ---------------------------------------------------------------------------
# Behavioural self-check
# ---------------------------------------------------------------------------

# (equity, current, pending, delta, expected_permitted, label)
_CANARIES: Final[tuple[tuple[int, int, int, int, bool, str], ...]] = (
    (1_000_000, 0, 0, 1_000_000, True, "exactly 1.0000x is allowed"),
    (1_000_000, 0, 0, 1_000_100, False, "1.0001x is rejected"),
    (1_000_000, 0, 0, 1_010_000, False, "1.01x is rejected"),
    (1_000_000, 0, 0, 1_100_000, False, "1.1x is rejected"),
    (1_000_000, 0, 0, 2_000_000, False, "2x is rejected"),
    (1_000_000, 0, 0, 1_000_001, False, "one micro-dollar over is rejected"),
    (1_000_000, 900_000, 0, 100_001, False, "pending+current+delta over the line is rejected"),
    (1_000_000, 500_000, 500_000, 1, False, "working orders consume headroom"),
    (0, 0, 0, 1, False, "zero equity blocks any increase"),
    (-5, 0, 0, 1, False, "negative equity blocks any increase"),
    (1_000_000, 2_000_000, 0, -500_000, True, "strict reduction allowed while over the line"),
    (1_000_000, 2_000_000, 0, 1, False, "increase forbidden while over the line"),
)


class InvariantCompromised(RuntimeError):
    """The leverage invariant does not behave as specified. Trading must stop.

    Not a ``RiskRejection``: this is not a rejected order, it is a compromised
    safety system, and it deliberately does not inherit from anything a generic
    ``except RiskRejection`` handler would catch.
    """


def assert_invariant_intact() -> None:
    """Verify the invariant still behaves exactly as specified.

    Called before every enforcement decision and at engine start-up. Cost is a
    dozen integer comparisons -- nanoseconds -- which buys detection of any
    monkeypatch, plugin, or accidental edit that weakened the rule.
    """
    if MAX_PERP_LEVERAGE != Decimal("1.00") or MAX_PERP_LEVERAGE_BASIS_POINTS != 10_000:
        raise InvariantCompromised(
            f"MAX_PERP_LEVERAGE has been altered to {MAX_PERP_LEVERAGE!r}; refusing to trade"
        )

    for equity, current, pending, delta, expected, label in _CANARIES:
        result = evaluate_perp_leverage(
            equity_micros=equity,
            current_exposure_micros=current,
            pending_exposure_micros=pending,
            requested_delta_micros=delta,
        )
        if result.permitted is not expected:
            raise InvariantCompromised(
                f"leverage invariant self-check failed -- {label}: expected "
                f"permitted={expected}, got permitted={result.permitted} "
                f"(reason={result.reason_code})"
            )

    # Malformed input must never be treated as permission.
    malformed = evaluate_perp_leverage(
        equity_micros=1_000_000,
        current_exposure_micros=0,
        pending_exposure_micros=0,
        requested_delta_micros="1",  # type: ignore[arg-type]
    )
    if malformed.permitted:
        raise InvariantCompromised(
            "leverage invariant self-check failed -- malformed input was permitted"
        )


def enforce_perp_leverage(
    *,
    equity_micros: int,
    current_exposure_micros: int,
    pending_exposure_micros: int,
    requested_delta_micros: int,
) -> LeverageAssessment:
    """Evaluate the invariant and raise :class:`LeverageViolation` if breached.

    This is the function the perpetual risk engine calls. It self-checks first,
    so a compromised invariant halts the platform instead of quietly approving
    orders.
    """
    assert_invariant_intact()

    assessment = evaluate_perp_leverage(
        equity_micros=equity_micros,
        current_exposure_micros=current_exposure_micros,
        pending_exposure_micros=pending_exposure_micros,
        requested_delta_micros=requested_delta_micros,
    )
    if not assessment.permitted:
        raise LeverageViolation(assessment.reason, detail=assessment.to_dict())
    return assessment
