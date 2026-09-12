"""A refusal repeated every cycle is one fact, not thirteen thousand rows.

A strategy that wants to trade a market it cannot trade proposes it again on the
next cycle, and the next. Every refusal was a decision row, a risk-event row and
a warning line: in three hours, 19,018 ORDER_RATE_LIMIT refusals for
event.passive_quote and 7,608 for event.model_drift, about 13,000 rows an hour
all saying the same sentence. That volume is what filled the writer's queue, and
anything that happened once was invisible inside it.

What must not happen is losing the refusal itself, or folding an approval: an
approval becomes an order, and an order needs the decision that authorised it.
"""

from __future__ import annotations

import time

from kat.common.enums import MarketType, RiskDecision, TradingEnvironment
from kat.engine.decision import _REPEAT_WINDOW_S, DecisionRecorder, TradeDecision


class _Writer:
    def __init__(self) -> None:
        self.rows: list[object] = []

    def enqueue(self, obj: object, *, critical: bool = False) -> bool:
        self.rows.append(obj)
        return True


def _decision(
    *, decision: RiskDecision = RiskDecision.REJECTED,
    reason: str = "ORDER_RATE_LIMIT",
    ticker: str = "KX-A",
    strategy: str = "event.passive_quote",
) -> TradeDecision:
    return TradeDecision(
        decision_id=f"dec_{time.monotonic_ns()}",
        market_type=MarketType.EVENT,
        ticker=ticker,
        decision=decision,
        reason_code=reason,
        reason="because",
        strategy_id=strategy,
        side="yes",
        environment=TradingEnvironment.PAPER,
    )


def _recorder() -> tuple[DecisionRecorder, _Writer]:
    writer = _Writer()
    return DecisionRecorder(writer), writer  # type: ignore[arg-type]


class TestRepeatedRejectionsAreFolded:
    def test_the_first_is_written_and_the_rest_are_counted(self):
        recorder, writer = _recorder()
        for _ in range(50):
            recorder.record(_decision())

        # One decision row and one risk-event row, for fifty refusals.
        assert len(writer.rows) == 2
        assert recorder.rejected == 50, "the count is unaffected by how it is stored"
        assert recorder.collapsed == 49
        assert recorder.stats()["collapsed"] == 49

    def test_the_row_after_the_window_says_how_many_it_stood_for(self):
        recorder, writer = _recorder()
        recorder.record(_decision())
        for _ in range(20):
            recorder.record(_decision())

        # The window closes; the next identical refusal is written again.
        key = next(iter(recorder._repeats))
        recorder._repeats[key][0] -= _REPEAT_WINDOW_S + 1
        recorder.record(_decision())

        rows = [r for r in writer.rows if hasattr(r, "risk_checks")]
        assert len(rows) == 2
        assert rows[1].risk_checks["repeats_folded"] == 20

    def test_a_different_reason_is_a_different_fact(self):
        recorder, writer = _recorder()
        recorder.record(_decision(reason="ORDER_RATE_LIMIT"))
        recorder.record(_decision(reason="INSUFFICIENT_CAPITAL"))
        recorder.record(_decision(reason="ORDER_RATE_LIMIT"))

        assert len([r for r in writer.rows if hasattr(r, "risk_checks")]) == 2

    def test_a_different_market_is_a_different_fact(self):
        recorder, writer = _recorder()
        recorder.record(_decision(ticker="KX-A"))
        recorder.record(_decision(ticker="KX-B"))

        assert len([r for r in writer.rows if hasattr(r, "risk_checks")]) == 2


class TestApprovalsAreNeverFolded:
    def test_every_approval_is_written(self):
        """It becomes an order, and the order needs the decision behind it."""
        recorder, writer = _recorder()
        for _ in range(10):
            recorder.record(_decision(decision=RiskDecision.APPROVED, reason="APPROVED"))

        rows = [r for r in writer.rows if hasattr(r, "risk_checks")]
        assert len(rows) == 10
        assert recorder.collapsed == 0

    def test_a_resize_is_written_too(self):
        recorder, writer = _recorder()
        for _ in range(5):
            recorder.record(_decision(decision=RiskDecision.RESIZED, reason="APPROVED"))
        assert len([r for r in writer.rows if hasattr(r, "risk_checks")]) == 5


class TestTheMapStaysBounded:
    def test_expired_keys_are_dropped(self):
        recorder, _ = _recorder()
        for n in range(4_100):
            recorder.record(_decision(ticker=f"KX-{n}"))
        for entry in recorder._repeats.values():
            entry[0] -= _REPEAT_WINDOW_S + 1
        recorder.record(_decision(ticker="KX-trigger"))

        assert len(recorder._repeats) < 4_000
