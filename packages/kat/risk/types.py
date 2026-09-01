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
from kat.common.fixedpoint import CC_PER_DOLLAR, micros_to_decimal

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

    #: Expected value of the round trip, **already net of expected costs**.
    #:
    #: The producer subtracts fees. ``kat.strategies.common`` builds this from
    #: ``Costing.net_micros``, which is ``gross - fees`` over an entry *and* an
    #: exit. A consumer must therefore compare it to a floor directly and must
    #: not subtract fees again.
    #:
    #: The unit was previously undocumented, and the event risk engine took it
    #: for a gross figure and subtracted an entry fee from it. That charged fees
    #: twice -- once round-trip by the strategy, once more by the engine -- and
    #: rejected trades whose real expected value was positive. Anything reading
    #: this field is relying on the sentence above.
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


    def edge_cc(self, market_price_cc: int | None = None) -> int | None:
        """Signed edge in centicents: what it is worth, less what we pay.

        Signed so a YES buy below fair value and a NO buy above it both come out
        positive. The NO side is the half that is easy to get wrong: a NO
        contract is worth ``1 - fair_value``, not ``fair_value``, so the same
        subtraction on both sides misprices every NO trade by the whole dollar.

        This lives on the intent because it was previously written twice -- once
        in the risk engine, which handled the NO side, and once in the decision
        recorder, which did not. The gate approved a Chelsea NO at +492, which
        was right, and the audit trail recorded the identical trade at -7,492,
        which was the YES formula applied to a NO order. Both numbers described
        the same order. Only one of them was ever read by a person, and it said
        the platform was buying things worth ten cents for eighty-five.
        """
        if self.fair_value_cc is None:
            return None
        reference = self.limit_price_cc if self.limit_price_cc else market_price_cc
        if reference is None:
            return None
        if self.outcome_side is OutcomeSide.NO:
            return (CC_PER_DOLLAR - self.fair_value_cc) - reference
        return self.fair_value_cc - reference


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
