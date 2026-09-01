"""The exchange's ``fee_multiplier`` is a scale, not the rate.

``GET /series/{ticker}`` returns ``fee_type: "quadratic"`` and
``fee_multiplier: 1`` -- verified against production, where all eighteen series
sampled returned exactly that. The ``1`` means "charged at the standard rate",
scaling Kalshi's published base coefficient of 0.07. It is not the coefficient
itself: the fee schedule module says so in its own opening docstring, that the
base rate is in the published PDF and never in the API.

``FeeSchedule.from_series`` takes the coefficient, because that is what every
caller in this codebase passes. Handing it the exchange's value instead would
build the model ``1.0 * C * P * (1 - P)`` -- 25 cents on a $0.50 contract where
Kalshi charges 1.75 -- and mark it *exact*, so nothing downstream would add a
safety margin or otherwise doubt it. Every expected value in the platform would
go negative at once and the strategies would stop trading for a reason no log
would name.

Nothing had wired the series endpoint up yet, so the trap was never sprung.
These tests keep it that way.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kat.kalshi.fees import (
    BASE_COEFFICIENT,
    FeeSchedule,
    estimate_fee_micros,
    realised_fee_micros,
)

#: Exactly what production returns, for every series sampled.
LIVE_PAYLOAD = {"fee_type": "quadratic", "fee_multiplier": 1}

#: Kalshi's published general trading fee peaks at a $0.50 contract and is
#: 0.07 * 1 * 0.5 * 0.5 = $0.0175 per contract.
MAX_FEE_PER_CONTRACT_MICROS = 17_500


class TestTheExchangePayloadIsReadAsAScale:
    def test_a_standard_series_yields_the_base_coefficient(self):
        schedule = FeeSchedule.from_exchange_series(LIVE_PAYLOAD)

        assert schedule.multiplier == BASE_COEFFICIENT
        assert schedule.fee_type == "quadratic"
        assert schedule.exact

    def test_the_fee_matches_kalshis_published_maximum(self):
        """One contract at $0.50 is the worst case, and it is 1.75 cents."""
        schedule = FeeSchedule.from_exchange_series(LIVE_PAYLOAD)

        fee = estimate_fee_micros(price_cc=5_000, qty_qc=100, schedule=schedule)
        assert fee.fee_micros == MAX_FEE_PER_CONTRACT_MICROS

    def test_reading_the_scale_as_the_coefficient_is_fourteen_times_too_much(self):
        """The bug being prevented, stated as a number."""
        correct = estimate_fee_micros(
            price_cc=5_000,
            qty_qc=100,
            schedule=FeeSchedule.from_exchange_series(LIVE_PAYLOAD),
        ).fee_micros
        naive = estimate_fee_micros(
            price_cc=5_000,
            qty_qc=100,
            schedule=FeeSchedule.from_series("quadratic", 1),
        ).fee_micros

        assert naive == 250_000
        assert naive / correct == pytest.approx(1 / float(BASE_COEFFICIENT), rel=1e-6)

    @pytest.mark.parametrize("scale", [Decimal("0.5"), 1, 2, "1.0"])
    def test_a_scale_is_applied_to_the_base_rate(self, scale):
        schedule = FeeSchedule.from_exchange_series(
            {"fee_type": "quadratic", "fee_multiplier": scale}
        )
        assert schedule.multiplier == BASE_COEFFICIENT * Decimal(str(scale))
        assert schedule.exact


class TestAnUnusablePayloadStaysInexact:
    """A schedule we cannot read must keep its safety margin, not lose it."""

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {},
            {"fee_type": "quadratic"},
            {"fee_type": "quadratic", "fee_multiplier": None},
            {"fee_type": "quadratic", "fee_multiplier": 0},
            {"fee_type": "quadratic", "fee_multiplier": -1},
            {"fee_type": "quadratic", "fee_multiplier": "not a number"},
        ],
    )
    def test_it_falls_back_to_the_base_rate_and_is_marked_inexact(self, payload):
        schedule = FeeSchedule.from_exchange_series(payload)

        assert schedule.multiplier == BASE_COEFFICIENT
        assert not schedule.exact

    def test_an_inexact_schedule_still_pads_the_estimate(self):
        """Losing `exact` would silently remove the margin, which is the point."""
        unknown = FeeSchedule.from_exchange_series(None)
        known = FeeSchedule.from_exchange_series(LIVE_PAYLOAD)

        padded = estimate_fee_micros(price_cc=5_000, qty_qc=100, schedule=unknown)
        exact = estimate_fee_micros(price_cc=5_000, qty_qc=100, schedule=known)
        assert padded.fee_micros > exact.fee_micros
        # ...but a charge is never padded, whatever we know.
        assert realised_fee_micros(
            price_cc=5_000, qty_qc=100, schedule=unknown
        ) == realised_fee_micros(price_cc=5_000, qty_qc=100, schedule=known)

    def test_the_fee_type_survives_an_unreadable_multiplier(self):
        schedule = FeeSchedule.from_exchange_series(
            {"fee_type": "quadratic_with_maker_fees", "fee_multiplier": None}
        )
        assert schedule.fee_type == "quadratic_with_maker_fees"
        assert schedule.maker_multiplier == Decimal("0.25")


class TestOverridesStillApply:
    def test_an_event_can_override_the_series_type(self):
        schedule = FeeSchedule.from_exchange_series(
            LIVE_PAYLOAD, fee_type_override="flat"
        )
        assert schedule.fee_type == "flat"
        assert schedule.multiplier == BASE_COEFFICIENT

    def test_an_override_scale_is_also_scaled_not_substituted(self):
        schedule = FeeSchedule.from_exchange_series(
            LIVE_PAYLOAD, fee_multiplier_override=2
        )
        assert schedule.multiplier == BASE_COEFFICIENT * 2
