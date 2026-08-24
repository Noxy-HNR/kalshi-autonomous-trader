"""The 1.00x perpetual leverage invariant.

These tests are the reason the rest of the platform can be trusted. They assert
that the limit holds at the boundary, that it cannot be talked around from any
layer of the system, and that missing or malformed information produces a
refusal rather than a guess.

If any test in this file fails, the platform must not trade.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kat.common.enums import BookSide, MarketType
from kat.common.errors import LeverageViolation
from kat.risk.invariants import (
    MAX_PERP_LEVERAGE,
    MAX_PERP_LEVERAGE_BASIS_POINTS,
    InvariantCompromised,
    assert_invariant_intact,
    enforce_perp_leverage,
    evaluate_perp_leverage,
    position_exposure_micros,
)
from kat.risk.types import OrderIntent

# $1.00 of equity, expressed in micro-dollars. Chosen so that the requested
# delta reads directly as a leverage multiple.
EQUITY = 1_000_000


def _at(multiple: str) -> int:
    """Exposure corresponding to a given leverage multiple against EQUITY."""
    return int(Decimal(multiple) * EQUITY)


def _evaluate(delta: int, *, equity: int = EQUITY, current: int = 0, pending: int = 0):
    return evaluate_perp_leverage(
        equity_micros=equity,
        current_exposure_micros=current,
        pending_exposure_micros=pending,
        requested_delta_micros=delta,
    )


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_exactly_one_times_is_allowed():
    """1.0000x is the maximum and is permitted, not rejected."""
    result = _evaluate(_at("1.0"))
    assert result.permitted is True
    assert result.projected_leverage == Decimal("1.000000")
    assert result.reason_code == "WITHIN_LIMIT"


@pytest.mark.parametrize(
    ("multiple", "label"),
    [
        ("1.0001", "1.0001x"),
        ("1.01", "1.01x"),
        ("1.1", "1.1x"),
        ("2", "2x"),
        ("3", "3x"),
        ("10", "10x"),
        ("100", "100x"),
    ],
)
def test_anything_above_one_times_is_rejected(multiple: str, label: str):
    result = _evaluate(_at(multiple))
    assert result.permitted is False, f"{label} must be rejected"
    assert result.reason_code == "LEVERAGE_WOULD_EXCEED_1X"


def test_one_micro_dollar_over_the_line_is_rejected():
    """The limit is exact: there is no tolerance band, not even a rounding one."""
    assert _evaluate(EQUITY).permitted is True
    assert _evaluate(EQUITY + 1).permitted is False


def test_the_check_is_exact_integer_arithmetic_not_float():
    """A value that float arithmetic would round to exactly 1.0 is still rejected.

    0.1 + 0.2 == 0.30000000000000004 in binary floating point. A float pipeline
    can therefore compute a leverage of 1.0 for a position that is truly over.
    Our comparison is integer, so this is impossible.
    """
    equity = 300_000  # $0.30
    # $0.10 + $0.20 in micro-dollars is exactly $0.30 -- allowed.
    assert _evaluate(200_000, equity=equity, current=100_000).permitted is True
    # One micro-dollar more is not.
    assert _evaluate(200_001, equity=equity, current=100_000).permitted is False


# ---------------------------------------------------------------------------
# Missing and malformed information
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["1", 1.0, None, True, False, Decimal("1"), [], {}, object(), float("nan"), float("inf")],
)
def test_malformed_delta_is_never_permitted(bad: object):
    """Anything that is not an exact int is refused, including bool and Decimal."""
    result = evaluate_perp_leverage(
        equity_micros=EQUITY,
        current_exposure_micros=0,
        pending_exposure_micros=0,
        requested_delta_micros=bad,  # type: ignore[arg-type]
    )
    assert result.permitted is False
    assert result.reason_code == "MALFORMED_RISK_INPUT"


@pytest.mark.parametrize("bad", ["200", 200.0, None, Decimal("200")])
def test_malformed_equity_is_never_permitted(bad: object):
    result = evaluate_perp_leverage(
        equity_micros=bad,  # type: ignore[arg-type]
        current_exposure_micros=0,
        pending_exposure_micros=0,
        requested_delta_micros=1,
    )
    assert result.permitted is False
    assert result.reason_code == "MALFORMED_RISK_INPUT"


@pytest.mark.parametrize("equity", [0, -1, -1_000_000])
def test_missing_or_non_positive_equity_blocks_every_increase(equity: int):
    """Unknown equity is not treated as "probably fine"."""
    result = _evaluate(1, equity=equity)
    assert result.permitted is False
    assert result.reason_code == "EQUITY_UNKNOWN_OR_NON_POSITIVE"


def test_negative_absolute_exposure_is_refused():
    """Absolute exposure cannot be negative; such input is corrupt, not small."""
    result = _evaluate(1, current=-1)
    assert result.permitted is False
    assert result.reason_code == "NEGATIVE_ABSOLUTE_EXPOSURE"


def test_negative_mark_price_cannot_produce_exposure():
    with pytest.raises(LeverageViolation):
        position_exposure_micros(1000, -1)


@pytest.mark.parametrize("bad", [1.0, "10", None, True])
def test_position_exposure_rejects_non_integer_inputs(bad: object):
    with pytest.raises(LeverageViolation):
        position_exposure_micros(bad, 5000)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Working orders consume headroom
# ---------------------------------------------------------------------------


def test_pending_orders_count_against_the_limit():
    """An order resting on the book is an order that can fill."""
    # Half the equity is already committed, half is locked by working orders.
    assert _evaluate(1, current=500_000, pending=500_000).permitted is False


def test_headroom_accounts_for_current_and_pending():
    result = _evaluate(0, current=300_000, pending=200_000)
    assert result.headroom_micros == 500_000


def test_headroom_never_goes_negative():
    result = _evaluate(0, current=2_000_000, pending=500_000)
    assert result.headroom_micros == 0


# ---------------------------------------------------------------------------
# De-risking is permitted; it is not a bypass
# ---------------------------------------------------------------------------


def test_strict_reduction_allowed_when_already_over_the_line():
    """A mark move can push leverage over with no order of ours involved.

    Forbidding the only orders that fix it would be the unsafe choice.
    """
    result = _evaluate(-500_000, current=2_000_000)
    assert result.permitted is True
    assert result.reason_code == "PERMITTED_EXPOSURE_REDUCTION"
    assert result.is_reduction is True


def test_increase_still_forbidden_when_already_over_the_line():
    assert _evaluate(1, current=2_000_000).permitted is False


def test_reduction_permitted_even_when_equity_is_unknown():
    """Flattening into an unknown-equity state lowers risk and must be possible."""
    result = _evaluate(-500_000, equity=0, current=2_000_000)
    assert result.permitted is True
    assert result.reason_code == "PERMITTED_EXPOSURE_REDUCTION"


def test_zero_delta_is_not_treated_as_a_reduction():
    """A no-op order gets no special treatment when already over the limit."""
    result = _evaluate(0, current=2_000_000)
    assert result.permitted is False


def test_over_stated_reduction_cannot_drive_exposure_negative():
    result = _evaluate(-10_000_000, current=1_000_000)
    assert result.projected_exposure_micros == 0
    assert result.permitted is True


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_the_declared_maximum_is_one():
    assert Decimal("1.00") == MAX_PERP_LEVERAGE
    assert MAX_PERP_LEVERAGE_BASIS_POINTS == 10_000


def test_self_check_passes_on_an_intact_build():
    # Returns None and, crucially, does not raise.
    assert assert_invariant_intact() is None


def test_self_check_detects_a_weakened_evaluation(monkeypatch: pytest.MonkeyPatch):
    """Replacing the evaluator with a permissive one must be caught.

    This is the scenario the canaries exist for: a plugin, a test double, or a
    careless edit that makes the check always say yes.
    """
    import kat.risk.invariants as inv

    def always_allow(**kwargs: object) -> inv.LeverageAssessment:
        return inv.LeverageAssessment(
            equity_micros=1,
            current_exposure_micros=0,
            pending_exposure_micros=0,
            requested_delta_micros=0,
            projected_exposure_micros=0,
            permitted=True,
            reason_code="WITHIN_LIMIT",
            reason="compromised",
        )

    monkeypatch.setattr(inv, "evaluate_perp_leverage", always_allow)
    with pytest.raises(InvariantCompromised):
        inv.assert_invariant_intact()


def test_self_check_detects_an_altered_constant(monkeypatch: pytest.MonkeyPatch):
    import kat.risk.invariants as inv

    monkeypatch.setattr(inv, "MAX_PERP_LEVERAGE", Decimal("2.00"))
    with pytest.raises(InvariantCompromised):
        inv.assert_invariant_intact()


def test_enforce_runs_the_self_check_before_deciding(monkeypatch: pytest.MonkeyPatch):
    """A compromised invariant halts rather than approving orders."""
    import kat.risk.invariants as inv

    monkeypatch.setattr(inv, "MAX_PERP_LEVERAGE", Decimal("5.00"))
    with pytest.raises(InvariantCompromised):
        inv.enforce_perp_leverage(
            equity_micros=EQUITY,
            current_exposure_micros=0,
            pending_exposure_micros=0,
            requested_delta_micros=1,
        )


def test_enforce_raises_leverage_violation_with_full_arithmetic():
    with pytest.raises(LeverageViolation) as exc:
        enforce_perp_leverage(
            equity_micros=EQUITY,
            current_exposure_micros=0,
            pending_exposure_micros=0,
            requested_delta_micros=EQUITY + 1,
        )
    detail = exc.value.detail
    assert detail["reason_code"] == "LEVERAGE_WOULD_EXCEED_1X"
    # Six decimals, so a rejection at 1.000001x does not render as "1.0000".
    assert detail["projected_leverage"] == "1.000001"
    assert detail["max_leverage"] == "1.00"


# ---------------------------------------------------------------------------
# There is no configuration route to a higher limit
# ---------------------------------------------------------------------------


def test_settings_expose_no_leverage_field():
    """Configuration must not contain anything that looks like a leverage knob."""
    from kat.config.settings import AppSettings, PerpRiskSettings

    for model in (PerpRiskSettings, AppSettings):
        for field_name in model.model_fields:
            assert "leverage" not in field_name.lower(), (
                f"{model.__name__}.{field_name} looks like a configurable leverage "
                "limit; the 1.00x ceiling must not be configurable"
            )


def test_configured_exposure_fraction_cannot_exceed_one():
    """Even the tightening knob is clamped, so it can never loosen the limit."""
    from kat.config.settings import PerpRiskSettings

    settings = PerpRiskSettings(max_exposure_pct_equity=Decimal("5.0"))
    assert settings.max_exposure_pct_equity <= Decimal("1.00")

    settings = PerpRiskSettings(max_position_pct_equity=Decimal("99"))
    assert settings.max_position_pct_equity <= Decimal("1.00")


def test_no_leverage_parameter_reaches_the_evaluator():
    """The evaluator's signature contains no maximum to supply.

    If a caller could pass a limit, some caller eventually would.
    """
    import inspect

    params = set(inspect.signature(evaluate_perp_leverage).parameters)
    assert params == {
        "equity_micros",
        "current_exposure_micros",
        "pending_exposure_micros",
        "requested_delta_micros",
    }


# ---------------------------------------------------------------------------
# The invariant reaches real orders through the risk engine
# ---------------------------------------------------------------------------


def _perp_intent(qty_qc: int, *, price_cc: int = 5000, side: BookSide = BookSide.BID):
    return OrderIntent(
        market_type=MarketType.PERP,
        ticker="TEST-PERP",
        qty_qc=qty_qc,
        limit_price_cc=price_cc,
        book_side=side,
        strategy_id="test-strategy",
    )


def test_risk_engine_permits_exactly_one_times(perp_risk, perp_state, liquid_perp_book, now_ms):
    """$200 equity at a $0.50 mark is exactly 400 contracts."""
    perp_state.positions["TEST-PERP"] = _flat_position()
    verdict = perp_risk.evaluate(
        _perp_intent(40_000), perp_state, book=liquid_perp_book, now_ms=now_ms
    )
    assert verdict.approved is True
    assert verdict.leverage["projected_leverage"] == "1.000000"


@pytest.mark.parametrize(
    ("qty_qc", "label"),
    [(40_001, "1.000025x"), (40_400, "1.01x"), (44_000, "1.1x"), (80_000, "2x")],
)
def test_risk_engine_blocks_above_one_times(
    perp_risk, perp_state, liquid_perp_book, now_ms, qty_qc: int, label: str
):
    perp_state.positions["TEST-PERP"] = _flat_position()
    verdict = perp_risk.evaluate(
        _perp_intent(qty_qc), perp_state, book=liquid_perp_book, now_ms=now_ms
    )
    assert verdict.approved is False, f"{label} must be blocked"
    assert verdict.reason_code == "LEVERAGE_WOULD_EXCEED_1X"


def test_risk_engine_blocks_when_equity_is_unknown(perp_risk, perp_state, liquid_perp_book, now_ms):
    perp_state.equity_is_known = False
    perp_state.equity_micros = 0
    perp_state.positions["TEST-PERP"] = _flat_position()
    verdict = perp_risk.evaluate(
        _perp_intent(100), perp_state, book=liquid_perp_book, now_ms=now_ms
    )
    assert verdict.approved is False
    assert verdict.reason_code == "EQUITY_UNKNOWN_OR_NON_POSITIVE"


def test_risk_engine_blocks_on_stale_account_state(perp_risk, perp_state, liquid_perp_book, now_ms):
    """Leverage computed from stale equity is not a leverage check at all."""
    perp_state.updated_at_ms = now_ms - 10_000_000
    verdict = perp_risk.evaluate(
        _perp_intent(100), perp_state, book=liquid_perp_book, now_ms=now_ms
    )
    assert verdict.approved is False
    assert verdict.reason_code == "STALE_ACCOUNT_STATE"


def test_risk_engine_blocks_without_a_mark_price(perp_risk, perp_state, now_ms):
    """No mark means no exposure figure, which means no leverage check."""
    verdict = perp_risk.evaluate(_perp_intent(100), perp_state, book=None, now_ms=now_ms)
    assert verdict.approved is False
    assert verdict.reason_code == "NO_MARK_PRICE"


def test_exchange_reported_exposure_binds_when_larger(
    perp_risk, perp_state, liquid_perp_book, now_ms
):
    """If we missed a fill, Kalshi's larger number is the one that counts."""
    perp_state.positions["TEST-PERP"] = _flat_position()
    perp_state.exchange_total_notional_micros = 190_000_000  # $190 of $200
    verdict = perp_risk.evaluate(
        _perp_intent(10_000), perp_state, book=liquid_perp_book, now_ms=now_ms
    )  # +$50 -> $240
    assert verdict.approved is False
    assert verdict.reason_code == "LEVERAGE_WOULD_EXCEED_1X"
    assert verdict.leverage["current_exposure_usd"] == "190.000000"


def test_de_risking_permitted_while_over_the_limit(perp_risk, perp_state, liquid_perp_book, now_ms):
    from kat.risk.perp_risk import PerpPositionState

    # 500 contracts at $0.50 is $250 of notional against $200 of equity.
    perp_state.positions["TEST-PERP"] = PerpPositionState(
        ticker="TEST-PERP", position_qc=50_000, mark_price_cc=5000
    )
    reduce = _perp_intent(10_000, side=BookSide.ASK)
    reduce.reduce_only = True
    verdict = perp_risk.evaluate(reduce, perp_state, book=liquid_perp_book, now_ms=now_ms)
    assert verdict.approved is True

    increase = _perp_intent(100)
    assert perp_risk.evaluate(
        increase, perp_state, book=liquid_perp_book, now_ms=now_ms
    ).approved is False


def test_shorts_consume_exposure_exactly_like_longs(perp_risk, perp_state, liquid_perp_book, now_ms):
    """A short is not a negative position that offsets the limit."""
    perp_state.positions["TEST-PERP"] = _flat_position()
    verdict = perp_risk.evaluate(
        _perp_intent(40_001, side=BookSide.ASK), perp_state, book=liquid_perp_book, now_ms=now_ms
    )
    assert verdict.approved is False
    assert verdict.reason_code == "LEVERAGE_WOULD_EXCEED_1X"


def _flat_position():
    from kat.risk.perp_risk import PerpPositionState

    return PerpPositionState(ticker="TEST-PERP", position_qc=0, mark_price_cc=5000)
