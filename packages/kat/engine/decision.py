"""Decision records -- the answer to "why did this trade happen?".

A decision record is written for **every** considered trade, including the ones
that were rejected. That is deliberate: "why did it not trade?" is asked at least
as often as "why did it trade?", and a system that only records its actions
cannot answer the more common question.

Each record carries the full chain::

    market tick -> features -> model -> strategy signal -> portfolio sizing
      -> risk verdict -> order -> fill

so a fill can be walked back to the tick that caused it, and a rejection can be
explained without re-running anything. For perpetuals the record includes the
complete leverage arithmetic -- equity, current exposure, pending exposure,
requested delta, projected exposure and the resulting ratio -- so a leverage
block is self-explanatory in the audit trail.

Records are written through the async writer as *critical* rows, so they are
never dropped under load, and they are logged in full if a write is ever
abandoned.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from kat.common.enums import MarketType, RiskDecision, TradingEnvironment
from kat.common.fixedpoint import micros_to_decimal
from kat.common.ids import current_correlation_id, new_decision_id
from kat.db.models import Decision as DecisionRow
from kat.db.models import RiskEvent
from kat.db.writer import AsyncWriter
from kat.observability.logging import get_logger
from kat.risk.types import OrderIntent, RiskVerdict

log = get_logger(__name__)

__all__ = ["DecisionRecorder", "TradeDecision", "git_commit"]


@lru_cache(maxsize=1)
def git_commit() -> str | None:
    """Short git commit of the running code, for reproducibility.

    Cached: this shells out once per process, never per decision. Returns
    ``None`` outside a git checkout rather than raising, because a packaged
    deployment is a legitimate way to run and should not fail on a missing
    ``.git``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


@dataclass(slots=True)
class TradeDecision:
    """One complete, inspectable trading decision."""

    decision_id: str
    market_type: MarketType
    ticker: str
    decision: RiskDecision
    reason_code: str
    reason: str

    correlation_id: str | None = None
    strategy_id: str | None = None
    strategy_version: str | None = None
    model_id: str | None = None
    model_version: str | None = None

    prediction: float | None = None
    market_price_cc: int | None = None
    fair_value_cc: int | None = None
    edge_cc: int | None = None
    confidence: float | None = None
    expected_value_micros: int | None = None

    side: str | None = None
    requested_qty_qc: int | None = None
    approved_qty_qc: int | None = None
    limit_price_cc: int | None = None

    current_exposure_micros: int | None = None
    projected_exposure_micros: int | None = None
    estimated_fees_micros: int | None = None
    expected_slippage_cc: int | None = None

    leverage_assessment: dict[str, Any] | None = None
    risk_checks: dict[str, Any] | None = None
    features: dict[str, Any] | None = None
    llm_signal_id: str | None = None

    order_id: str | None = None
    execution_result: dict[str, Any] | None = None

    environment: TradingEnvironment = TradingEnvironment.PAPER
    app_version: str | None = None
    commit: str | None = None

    # Populated lazily by the recorder; kept off the row itself.
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def approved(self) -> bool:
        return self.decision in (RiskDecision.APPROVED, RiskDecision.RESIZED)

    def to_row(self) -> DecisionRow:
        return DecisionRow(
            decision_id=self.decision_id,
            correlation_id=self.correlation_id,
            market_type=self.market_type.value,
            ticker=self.ticker,
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            model_id=self.model_id,
            model_version=self.model_version,
            prediction=self.prediction,
            market_price_cc=self.market_price_cc,
            fair_value_cc=self.fair_value_cc,
            edge_cc=self.edge_cc,
            confidence=self.confidence,
            expected_value_micros=self.expected_value_micros,
            side=self.side,
            requested_qty_qc=self.requested_qty_qc,
            approved_qty_qc=self.approved_qty_qc,
            limit_price_cc=self.limit_price_cc,
            current_exposure_micros=self.current_exposure_micros,
            projected_exposure_micros=self.projected_exposure_micros,
            estimated_fees_micros=self.estimated_fees_micros,
            expected_slippage_cc=self.expected_slippage_cc,
            leverage_assessment=self.leverage_assessment,
            decision=self.decision.value,
            reason_code=self.reason_code,
            reason=self.reason,
            risk_checks=self.risk_checks,
            features=self.features,
            llm_signal_id=self.llm_signal_id,
            order_id=self.order_id,
            execution_result=self.execution_result,
            environment=self.environment.value,
            app_version=self.app_version,
            git_commit=self.commit,
        )

    def summary(self) -> str:
        """One-line human summary, as shown in the dashboard audit list."""
        qty = (self.approved_qty_qc or 0) / 100
        verb = {
            RiskDecision.APPROVED: "approved",
            RiskDecision.RESIZED: f"resized to {qty:g}ct",
            RiskDecision.REJECTED: "rejected",
        }[self.decision]
        price = f"@{self.limit_price_cc / 10_000:.4f}" if self.limit_price_cc else ""
        return f"{self.ticker} {self.side or ''} {verb} {price} -- {self.reason_code}".strip()


class DecisionRecorder:
    """Builds and persists decision records.

    Holds no trading state and makes no trading choices -- it observes what the
    pipeline decided and writes it down. Keeping it passive means a bug here can
    lose an audit record but cannot change a trade.
    """

    def __init__(
        self,
        writer: AsyncWriter | None,
        *,
        environment: TradingEnvironment = TradingEnvironment.PAPER,
        app_version: str | None = None,
    ) -> None:
        self.writer = writer
        self.environment = environment
        self.app_version = app_version
        self.commit = git_commit()
        self.recorded = 0
        self.rejected = 0

    def build(
        self,
        verdict: RiskVerdict,
        *,
        market_price_cc: int | None = None,
        prediction: float | None = None,
        model_id: str | None = None,
        model_version: str | None = None,
        features: dict[str, Any] | None = None,
        llm_signal_id: str | None = None,
    ) -> TradeDecision:
        """Assemble a record from a risk verdict and its surrounding context."""
        intent: OrderIntent = verdict.intent
        return TradeDecision(
            decision_id=intent.decision_id or new_decision_id(),
            market_type=intent.market_type,
            ticker=intent.ticker,
            decision=verdict.decision,
            reason_code=verdict.reason_code,
            reason=verdict.reason,
            correlation_id=intent.correlation_id or current_correlation_id(),
            strategy_id=intent.strategy_id,
            strategy_version=intent.strategy_version,
            model_id=model_id,
            model_version=model_version,
            prediction=prediction,
            market_price_cc=market_price_cc,
            fair_value_cc=intent.fair_value_cc,
            edge_cc=_edge(intent, market_price_cc),
            confidence=intent.confidence,
            expected_value_micros=intent.expected_value_micros,
            side=_side(intent),
            requested_qty_qc=intent.qty_qc,
            approved_qty_qc=verdict.approved_qty_qc,
            limit_price_cc=intent.limit_price_cc,
            current_exposure_micros=verdict.current_exposure_micros,
            projected_exposure_micros=verdict.projected_exposure_micros,
            estimated_fees_micros=verdict.estimated_fees_micros,
            expected_slippage_cc=verdict.expected_slippage_cc,
            leverage_assessment=verdict.leverage,
            risk_checks={"checks": [c.to_dict() for c in verdict.checks]},
            features=features,
            llm_signal_id=llm_signal_id,
            environment=self.environment,
            app_version=self.app_version,
            commit=self.commit,
        )

    def record(self, decision: TradeDecision) -> TradeDecision:
        """Persist a decision and log it. Safe to call with no writer configured."""
        self.recorded += 1
        if not decision.approved:
            self.rejected += 1

        if self.writer is not None:
            # Critical: a decision record is the audit trail, not telemetry.
            self.writer.enqueue(decision.to_row(), critical=True)

            # A rejection is also a risk event, so the Risk tab can show what
            # fired without scanning every decision.
            if not decision.approved:
                self.writer.enqueue(
                    RiskEvent(
                        correlation_id=decision.correlation_id,
                        decision_id=decision.decision_id,
                        market_type=decision.market_type.value,
                        ticker=decision.ticker,
                        strategy_id=decision.strategy_id,
                        reason_code=decision.reason_code,
                        message=decision.reason,
                        severity=(
                            "critical"
                            if decision.reason_code.startswith("LEVERAGE")
                            else "warning"
                        ),
                        detail={
                            "leverage": decision.leverage_assessment,
                            "checks": decision.risk_checks,
                        },
                    ),
                    critical=True,
                )

        log_fn = log.info if decision.approved else log.warning
        log_fn(
            "decision.recorded",
            decision_id=decision.decision_id,
            correlation_id=decision.correlation_id,
            market_type=decision.market_type.value,
            ticker=decision.ticker,
            strategy_id=decision.strategy_id,
            decision=decision.decision.value,
            reason_code=decision.reason_code,
            requested_qty_qc=decision.requested_qty_qc,
            approved_qty_qc=decision.approved_qty_qc,
            expected_value_usd=(
                str(micros_to_decimal(decision.expected_value_micros))
                if decision.expected_value_micros is not None
                else None
            ),
            projected_leverage=(
                (decision.leverage_assessment or {}).get("projected_leverage")
            ),
        )
        return decision

    def attach_execution(
        self,
        decision: TradeDecision,
        *,
        order_id: str,
        result: dict[str, Any],
    ) -> None:
        """Link the order and its outcome back onto the decision that produced it.

        Written as a second row for the same ``decision_id``, which the writer
        merges over the first -- so the record is complete once execution
        resolves, without holding the decision open while an order is in flight.
        """
        decision.order_id = order_id
        decision.execution_result = result
        if self.writer is not None:
            self.writer.enqueue(decision.to_row(), critical=True)
        log.info(
            "decision.executed",
            decision_id=decision.decision_id,
            order_id=order_id,
            result=result,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "recorded": self.recorded,
            "rejected": self.rejected,
            "approval_rate": (
                round(1 - self.rejected / self.recorded, 4) if self.recorded else None
            ),
        }


def _side(intent: OrderIntent) -> str | None:
    if intent.outcome_side is not None:
        return intent.outcome_side.value
    if intent.book_side is not None:
        return intent.book_side.value
    return None


def _edge(intent: OrderIntent, market_price_cc: int | None) -> int | None:
    """Signed edge in centicents. See :meth:`OrderIntent.edge_cc`.

    Kept as a thin wrapper rather than inlined, because the name is used at the
    call site and the indirection is where the old, unsigned version lived.
    """
    return intent.edge_cc(market_price_cc)
