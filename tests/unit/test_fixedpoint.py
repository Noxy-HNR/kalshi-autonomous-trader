"""Exact money arithmetic.

Every risk decision reduces to an integer comparison between two monetary
quantities, so these tests guard the layer that makes those integers exact.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kat.common.fixedpoint import (
    CC_PER_DOLLAR,
    MICROS_PER_DOLLAR,
    QC_PER_CONTRACT,
    FixedPointError,
    Rounding,
    format_count,
    format_dollars,
    micros_to_decimal,
    notional_micros,
    parse_count_qc,
    parse_dollars_micros,
    parse_price_cc,
    price_cc_to_decimal,
    qc_to_decimal,
)


class TestScales:
    def test_scale_constants(self):
        assert CC_PER_DOLLAR == 10_000
        assert QC_PER_CONTRACT == 100
        assert MICROS_PER_DOLLAR == 1_000_000

    def test_price_times_quantity_lands_exactly_on_the_money_scale(self):
        """1e-4 USD * 1e-2 contracts == 1e-6 USD, with no rounding step."""
        # 10 contracts at $0.56 is exactly $5.60.
        assert notional_micros(5600, 1000) == 5_600_000
        assert micros_to_decimal(notional_micros(5600, 1000)) == Decimal("5.600000")

    def test_notional_carries_the_sign_of_the_quantity(self):
        assert notional_micros(5000, -1000) == -5_000_000


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("0.5600", 5600), ("0.01", 100), ("1.0000", 10_000), ("0.0001", 1), ("0", 0)],
    )
    def test_parses_kalshi_price_strings(self, text: str, expected: int):
        assert parse_price_cc(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"), [("10.00", 1000), ("0.01", 1), ("2.50", 250), ("-5.00", -500)]
    )
    def test_parses_fixed_point_counts_including_fractional_and_signed(
        self, text: str, expected: int
    ):
        assert parse_count_qc(text) == expected

    def test_exact_mode_refuses_to_silently_lose_precision(self):
        """A price off the 1/10000 grid means a new tick structure or bad data."""
        with pytest.raises(FixedPointError):
            parse_price_cc("0.00005")

    def test_directional_rounding_lets_risk_round_against_the_trader(self):
        assert parse_dollars_micros("1.0000005", rounding=Rounding.FLOOR) == 1_000_000
        assert parse_dollars_micros("1.0000005", rounding=Rounding.CEIL) == 1_000_001

    def test_six_decimal_responses_are_representable(self):
        """Kalshi emits money with up to six decimals."""
        assert parse_dollars_micros("0.123456", rounding=Rounding.EXACT) == 123_456

    @pytest.mark.parametrize("bad", [True, False])
    def test_booleans_are_not_amounts(self, bad: bool):
        """bool subclasses int; accepting it would let True mean $0.0001."""
        with pytest.raises(FixedPointError):
            parse_price_cc(bad)

    @pytest.mark.parametrize("bad", ["", "   ", "abc", "1.2.3", None, [], {}])
    def test_junk_is_rejected(self, bad: object):
        with pytest.raises(FixedPointError):
            parse_price_cc(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_are_rejected(self, bad: float):
        with pytest.raises(FixedPointError):
            parse_dollars_micros(bad)

    def test_floats_are_read_as_written_not_as_binary_expansions(self):
        """0.07 must mean the literal 0.07, not 0.070000000000000006."""
        assert parse_dollars_micros(0.07) == 70_000

    def test_decimal_input_is_exact(self):
        assert parse_price_cc(Decimal("0.4200")) == 4200


class TestFormatting:
    def test_round_trips_through_kalshi_wire_format(self):
        for text in ("0.5600", "0.0100", "0.9999", "1.0000"):
            assert format_dollars(parse_price_cc(text)) == text

    def test_counts_render_with_two_decimals(self):
        assert format_count(1000) == "10.00"
        assert format_count(1) == "0.01"
        assert format_count(250) == "2.50"

    def test_decimal_conversions_are_exact(self):
        assert price_cc_to_decimal(4200) == Decimal("0.4200")
        assert qc_to_decimal(250) == Decimal("2.50")
        assert micros_to_decimal(5_600_000) == Decimal("5.600000")


class TestFloatWouldBeWrong:
    """Cases where a float pipeline produces a different answer than we do."""

    def test_repeated_addition_stays_exact(self):
        """0.1 added ten times is not 1.0 in binary floating point."""
        total = sum(parse_dollars_micros("0.10") for _ in range(10))
        assert total == 1_000_000
        assert micros_to_decimal(total) == Decimal("1.000000")

    def test_the_classic_case(self):
        """0.1 + 0.2 == 0.30000000000000004 as a float; exact for us."""
        a = parse_dollars_micros("0.1")
        b = parse_dollars_micros("0.2")
        assert a + b == parse_dollars_micros("0.3")
        assert (0.1 + 0.2) != 0.3  # the behaviour we are avoiding

    def test_a_leverage_comparison_that_float_would_get_wrong(self):
        """Exposure built from many small adds must not drift over the line."""
        equity = parse_dollars_micros("1.00")
        exposure = sum(parse_dollars_micros("0.01") for _ in range(100))
        assert exposure == equity  # exactly 1.00x, and therefore allowed
        assert exposure <= equity
