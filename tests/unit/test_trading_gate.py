"""The trading gate: what stops trading, and what may start it again.

The defect these pin: start-up refused to permit trading for any of seven
reasons, and the health loop re-permitted it five seconds later after checking
two of them. An engaged event kill switch, a failed reconciliation, or an
exchange we never reached were all silently cleared without anything having
changed. Observed live -- the platform logged
``app.degraded_trading_disabled reasons=['kill switch engaged: event ...']`` at
start-up and reported ``trading_permitted: true`` moments afterwards.

The rule now: refusal and recovery consult the same predicate, and a condition
no liveness probe can disprove stays latched until whoever set it clears it.
"""

from __future__ import annotations

import pytest

from kat.common.enums import KillSwitchScope
from kat.config.settings import AppSettings
from kat.engine.lifecycle import Application, Phase


@pytest.fixture
def app() -> Application:
    application = Application(AppSettings())
    # A started application, without a database or components behind it.
    application.phase = Phase.RUNNING
    application._trading_permitted = True
    return application


class _Switch:
    def __init__(self, scope: str, engaged: bool, subject: str = "", reason: str = "") -> None:
        self.scope = scope
        self.engaged = engaged
        self.subject = subject
        self.reason = reason


class _Switches:
    """Enough of KillSwitchManager for the gate to consult."""

    def __init__(self, *switches: _Switch) -> None:
        self._switches = list(switches)

    def all_statuses(self) -> list[_Switch]:
        return list(self._switches)

    def is_engaged(self, scope: KillSwitchScope, subject: str = "") -> bool:
        return any(s.engaged and s.scope == scope.value for s in self._switches)


class TestCurrentBlocks:
    def test_nothing_blocking_is_nothing_blocking(self, app):
        app.kill_switches = _Switches()
        assert app.current_blocks() == []

    def test_an_engaged_switch_of_any_scope_blocks(self, app):
        """Not only the global one.

        A domain switch is a deliberate instruction to stop trading that domain;
        reporting the application as freely trading while one is engaged is a
        lie to whoever engaged it.
        """
        for scope in ("global", "event", "perp", "risk"):
            app.kill_switches = _Switches(_Switch(scope, True, reason="test"))
            blocks = app.current_blocks()
            assert len(blocks) == 1, scope
            assert scope in blocks[0]

    def test_a_released_switch_stops_blocking_without_a_restart(self, app):
        app.kill_switches = _Switches(_Switch("event", False))
        assert app.current_blocks() == []

    def test_an_unhealthy_critical_component_blocks(self, app):
        app.kill_switches = _Switches()
        assert app.current_blocks(["event-market-data"]) == [
            "critical component event-market-data is unhealthy"
        ]


class TestLatchedBlocks:
    def test_a_latched_block_survives_a_clean_health_check(self, app):
        """The heart of it.

        "We do not know what positions we hold" is not something a liveness
        probe can disprove. It must not lapse because the next probe happened to
        come back green.
        """
        app.kill_switches = _Switches()
        app.add_block("reconcile:event", "event reconciliation failed")
        assert app.current_blocks() == ["event reconciliation failed"]
        # Every component healthy, every switch released -- still blocked.
        assert app.current_blocks(unhealthy_critical=[]) == ["event reconciliation failed"]

    def test_only_the_setter_can_clear_it(self, app):
        app.kill_switches = _Switches()
        app.add_block("kalshi", "Kalshi is unreachable: timeout")
        assert app.current_blocks()
        app.clear_block("kalshi")
        assert app.current_blocks() == []

    def test_clearing_an_unset_block_is_harmless(self, app):
        app.kill_switches = _Switches()
        app.clear_block("never-set")
        assert app.current_blocks() == []

    def test_latching_twice_reports_once(self, app):
        app.kill_switches = _Switches()
        app.add_block("kalshi", "Kalshi is unreachable: timeout")
        app.add_block("kalshi", "Kalshi is unreachable: timeout")
        assert app.current_blocks() == ["Kalshi is unreachable: timeout"]
        assert app.report.blocking_reasons.count("Kalshi is unreachable: timeout") == 1

    def test_blocks_appear_in_the_startup_report(self, app):
        """The dashboard reads this. It has to say why."""
        app.kill_switches = _Switches()
        app.add_block("reconcile:perp", "perpetual reconciliation failed")
        assert "perpetual reconciliation failed" in app.report.blocking_reasons


class TestTradingGate:
    def test_the_gate_is_shut_before_start_up_completes(self):
        application = Application(AppSettings())
        application.kill_switches = _Switches()
        assert application.trading_permitted is False

    def test_the_gate_is_shut_while_degraded(self, app):
        app.kill_switches = _Switches()
        app.phase = Phase.DEGRADED
        assert app.trading_permitted is False

    def test_the_global_switch_shuts_the_gate_directly(self, app):
        app.kill_switches = _Switches(_Switch("global", True, reason="halt"))
        assert app.trading_permitted is False

    def test_real_orders_need_the_gate_and_the_environment(self, app):
        """Two independent conditions, both required."""
        app.kill_switches = _Switches()
        assert app.trading_permitted is True
        # Paper is the default; real orders must still be refused.
        assert app.can_submit_real_orders is False
