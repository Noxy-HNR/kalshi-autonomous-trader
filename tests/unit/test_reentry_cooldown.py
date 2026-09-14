"""A market that was just exited refuses entries for a while, and never refuses exits.

Two strategies held opposite views of one market. news_reaction bought YES every
cycle; model_drift, whose forecast said the opposite, sold it back in the same
cycle. Neither was wrong by its own lights, and nothing in either strategy could
see the other, so the pair paid the spread once a second: 31 exits and 30
re-entries in two minutes across two markets, -$3.51, about $84 an hour.

The refusal lives in the risk engine because the loss is in the pair.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kat.common.enums import MarketType, OutcomeSide
from kat.config.settings import EventRiskSettings
from kat.risk.event_risk import EventAccountState, EventPositionState, EventRiskEngine
from kat.risk.types import OrderIntent

NOW_MS = 10_000_000_000
COOLDOWN_MS = EventRiskSettings().reentry_cooldown_s * 1000


def _state() -> EventAccountState:
    state = EventAccountState(balance_micros=200_000_000, updated_at_ms=NOW_MS)
    state.peak_equity_micros = state.equity_micros
    return state


def _intent(ticker: str = "X", *, side: OutcomeSide = OutcomeSide.YES, qty_qc: int = 100) -> OrderIntent:
    return OrderIntent(
        market_type=MarketType.EVENT, ticker=ticker, qty_qc=qty_qc,
        limit_price_cc=5_000, outcome_side=side, strategy_id="s",
    )


class TestEntriesWaitOutTheCooldown:
    def test_an_entry_right_after_an_exit_is_refused(self):
        state = _state()
        state.note_exit("X", NOW_MS - 1_000, keep_ms=COOLDOWN_MS)

        verdict = EventRiskEngine(EventRiskSettings()).evaluate(_intent(), state, now_ms=NOW_MS)
        assert verdict.reason_code == "REENTRY_COOLDOWN"

    def test_the_same_entry_is_allowed_once_the_cooldown_has_passed(self):
        state = _state()
        state.note_exit("X", NOW_MS - COOLDOWN_MS - 1_000, keep_ms=COOLDOWN_MS)

        verdict = EventRiskEngine(EventRiskSettings()).evaluate(_intent(), state, now_ms=NOW_MS)
        assert verdict.reason_code != "REENTRY_COOLDOWN"

    def test_another_market_is_not_affected(self):
        state = _state()
        state.note_exit("X", NOW_MS - 1_000, keep_ms=COOLDOWN_MS)

        verdict = EventRiskEngine(EventRiskSettings()).evaluate(_intent("Y"), state, now_ms=NOW_MS)
        assert verdict.reason_code != "REENTRY_COOLDOWN"


class TestExitsAreNeverRefused:
    def test_closing_a_position_during_the_cooldown_is_allowed(self):
        """A halted or cooling platform must still be able to get out."""
        state = _state()
        state.positions["X"] = EventPositionState(
            ticker="X", position_qc=100, avg_entry_cc=5_000, exposure_micros=500_000,
        )
        state.note_exit("X", NOW_MS - 1_000, keep_ms=COOLDOWN_MS)

        verdict = EventRiskEngine(EventRiskSettings()).evaluate(
            _intent(side=OutcomeSide.NO, qty_qc=100), state, now_ms=NOW_MS
        )
        assert verdict.reason_code != "REENTRY_COOLDOWN"


class TestTheRecordStaysBounded:
    def test_old_exits_are_dropped_as_new_ones_arrive(self):
        state = _state()
        for i in range(1_000):
            state.note_exit(f"OLD-{i}", NOW_MS - COOLDOWN_MS * 2, keep_ms=COOLDOWN_MS)
        state.note_exit("NEW", NOW_MS, keep_ms=COOLDOWN_MS)

        assert len(state.recent_exits_ms) <= 257
        assert "NEW" in state.recent_exits_ms


class TestTheCooldownCanOnlyBeLengthened:
    def test_it_cannot_be_set_below_the_default(self):
        with pytest.raises(ValidationError):
            EventRiskSettings(reentry_cooldown_s=60)

    def test_it_can_be_made_longer(self):
        assert EventRiskSettings(reentry_cooldown_s=7_200).reentry_cooldown_s == 7_200
