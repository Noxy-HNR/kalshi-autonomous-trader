"""Shared vocabulary for the risk layer.

The event and perpetual risk engines are entirely separate implementations --
they share no calculation -- but they must speak the same language to the
portfolio, execution, audit and dashboard layers. That language is defined here.

The order of the pipeline is fixed and one-directional::

    Strategy -> Portfolio/Allocation -> Risk Engine -> Execution Engine -> Kalshi

A strategy produces an :class:`OrderIntent`. It never produces an order. Only
the risk engine can turn an intent into something the execution engine will
accept, and it does so by returning a :class:`RiskVerdict` that is itself the
authorisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kat.common.enums import BookSide, MarketType, OutcomeSide, RiskDecision
from kat.common.fixedpoint import micros_to_decimal

__all__ = ["CheckResult", "OrderIntent", "RiskVerdict"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One named risk check and its outcome.

    Every check that ran is recorded, passing or failing, so the decision record
    shows what was verified and not merely what went wrong.
    """

    name: str
    passed: bool
    limit: str | None = None
    observed: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "limit": self.limit,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass(slots=True)
class OrderIntent:
    """What some upstream component would like to do.

    Deliberately not an order: it has no client order id, no exchange fields and
    no way to reach the network. It is a request for permission.
    """

    market_type: MarketType
    ticker: str
    qty_qc: int
    limit_price_cc: int

    # Event contracts express direction as an outcome side; perpetuals as a
    # book side. Exactly one is set, according to ``market_type``.
    outcome_side: OutcomeSide | None = None
    book_side: BookSide | None = None

    strategy_id: str | None = None
    strategy_version: str | None = None
    decision_id: str | None = None
    correlation_id: str | None = None

    reduce_only: bool = False
    post_only: bool = False
    # Set by an emergency-flatten path. Does NOT bypass any check: it only marks
    # intent, and the risk engine still verifies the order genuinely reduces
    # exposure before treating it as a reduction.
    is_flatten: bool = False

    expected_value_micros: int | None = None
    fair_value_cc: int | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def notional_micros(self) -> int:
        """Exact notional of this intent at its limit price."""
        return abs(self.qty_qc) * self.limit_price_cc

    @property
    def is_buy(self) -> bool:
        """True when this intent lifts offers.

        For event contracts a YES buy and a NO buy are both "buys" in the sense
        that matters here -- they both consume capital -- so direction is read
        from the book side the order will rest on.
        """
        if self.book_side is not None:
            return self.book_side is BookSide.BID
        return self.outcome_side is OutcomeSide.YES


@dataclass(slots=True)
class RiskVerdict:
    """The risk engine's authorisation, or its refusal.

    ``approved_qty_qc`` may be smaller than the requested size: resizing is a
    normal outcome, not an error, and is how position-size and exposure limits
    are applied without rejecting an otherwise good trade outright.
    """

    decision: RiskDecision
    intent: OrderIntent
    approved_qty_qc: int
    reason_code: str
    reason: str
    checks: list[CheckResult] = field(default_factory=list)

    # Perp-only. Present on every perpetual verdict, approved or not.
    leverage: dict[str, Any] | None = None

    estimated_fees_micros: int = 0
    expected_slippage_cc: int | None = None
    current_exposure_micros: int = 0
    projected_exposure_micros: int = 0

    @property
    def approved(self) -> bool:
        return self.decision in (RiskDecision.APPROVED, RiskDecision.RESIZED)

    @property
    def approved_notional_micros(self) -> int:
        return abs(self.approved_qty_qc) * self.intent.limit_price_cc

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        """Flat form for the decision record and the dashboard."""
        return {
            "decision": self.decision.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "requested_qty": str(Decimal(self.intent.qty_qc) / 100),
            "approved_qty": str(Decimal(self.approved_qty_qc) / 100),
            "estimated_fees_usd": str(micros_to_decimal(self.estimated_fees_micros)),
            "expected_slippage_cc": self.expected_slippage_cc,
            "current_exposure_usd": str(micros_to_decimal(self.current_exposure_micros)),
            "projected_exposure_usd": str(micros_to_decimal(self.projected_exposure_micros)),
            "checks": [c.to_dict() for c in self.checks],
            "leverage": self.leverage,
        }

    # -- constructors --------------------------------------------------------

    @classmethod
    def reject(
        cls,
        intent: OrderIntent,
        reason_code: str,
        reason: str,
        *,
        checks: list[CheckResult] | None = None,
        leverage: dict[str, Any] | None = None,
    ) -> RiskVerdict:
        return cls(
            decision=RiskDecision.REJECTED,
            intent=intent,
            approved_qty_qc=0,
            reason_code=reason_code,
            reason=reason,
            checks=checks or [],
            leverage=leverage,
        )
