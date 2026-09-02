"""Perpetuals must not be charged an event contract's fee.

The two instruments are priced on unrelated schedules. An event contract pays
``m * C * P * (1 - P)``, which peaks at 175 centicents a side on a fifty-cent
contract -- 350 basis points of notional. A perpetual pays a flat rate on
notional: 2bp to a maker, 4bp to a taker.

The simulator had one fee path and it was the quadratic one, so every perpetual
strategy in this codebase -- three of them, with a risk engine and an execution
engine behind them -- would have been costed about a hundred times too dearly.
That error runs in the direction that silently discards strategies that work,
which is the expensive direction and the quiet one.
"""

from __future__ import annotations

from decimal import Decimal

from kat.kalshi.fees import (
    FeeSchedule,
    PerpFeeSchedule,
    estimate_fee_micros,
    perp_fee_micros,
)
from kat.simulation.costs import CostModel

#: 100 contracts of a perpetual quoted at $8.23 -- $823 of notional.
PRICE_CC = 82_300
QTY_QC = 10_000
NOTIONAL_MICROS = 823_000_000

TIERS = {
    "maker_fee_rates": {"KXZECPERP": 0.0002},
    "taker_fee_rates": {"KXZECPERP": 0.0004},
}


class TestTheRateAppliesToNotional:
    def test_taker_pays_four_basis_points(self):
        schedule = PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP")
        fee = perp_fee_micros(price_cc=PRICE_CC, qty_qc=QTY_QC, schedule=schedule)
        assert fee == int(NOTIONAL_MICROS * 0.0004)

    def test_maker_pays_half_of_that(self):
        schedule = PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP")
        fee = perp_fee_micros(
            price_cc=PRICE_CC, qty_qc=QTY_QC, schedule=schedule, is_taker=False
        )
        assert fee == int(NOTIONAL_MICROS * 0.0002)

    def test_qty_times_price_is_already_micros(self):
        """No rescaling. Centi-contracts times ten-thousandths of a dollar is
        micros of notional, and a conversion factor here would be wrong by 100."""
        schedule = PerpFeeSchedule(taker_rate=Decimal("1"), exact=True)
        assert perp_fee_micros(
            price_cc=PRICE_CC, qty_qc=QTY_QC, schedule=schedule
        ) == NOTIONAL_MICROS

    def test_rounding_never_favours_us(self):
        schedule = PerpFeeSchedule(taker_rate=Decimal("0.0004"), exact=True)
        # One centi-contract at one centicent: a fee far below a single micro.
        assert perp_fee_micros(price_cc=1, qty_qc=1, schedule=schedule) == 1

    def test_nothing_traded_costs_nothing(self):
        assert perp_fee_micros(price_cc=PRICE_CC, qty_qc=0) == 0
        assert perp_fee_micros(price_cc=0, qty_qc=QTY_QC) == 0


class TestItRefusesToGuess:
    def test_a_market_absent_from_the_tiers_gets_no_maker_discount(self):
        """Inheriting another market's discount would be inventing a rate."""
        schedule = PerpFeeSchedule.from_fee_tiers(TIERS, "KXNOTLISTED")
        assert schedule.exact is False
        assert schedule.maker_rate == schedule.taker_rate

    def test_an_unparseable_rate_falls_back_rather_than_raising(self):
        bad = {"maker_fee_rates": {"X": "nonsense"}, "taker_fee_rates": {"X": "?"}}
        schedule = PerpFeeSchedule.from_fee_tiers(bad, "X")
        assert schedule.exact is False

    def test_an_empty_payload_is_survivable(self):
        assert PerpFeeSchedule.from_fee_tiers({}, "X").exact is False
        assert PerpFeeSchedule.from_fee_tiers(None, "X").exact is False


class TestTheSimulatorChargesTheRightOne:
    def test_a_perp_costs_two_orders_of_magnitude_less(self):
        """The whole point. At fifty cents the event fee is 350bp of notional
        and the perp taker fee is 4bp."""
        half_dollar, qty = 5_000, 10_000
        event = CostModel().fee_micros(
            price_cc=half_dollar, qty_qc=qty, is_taker=True
        )
        perp = CostModel(
            perp_fees=PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP")
        ).fee_micros(price_cc=half_dollar, qty_qc=qty, is_taker=True)
        assert perp < event / 50

    def test_the_event_path_is_untouched_when_no_perp_schedule_is_set(self):
        expected = estimate_fee_micros(
            price_cc=5_000, qty_qc=10_000, schedule=FeeSchedule(), is_taker=True
        ).fee_micros
        assert CostModel().fee_micros(
            price_cc=5_000, qty_qc=10_000, is_taker=True
        ) == expected

    def test_the_maker_discount_is_honoured_on_perps(self):
        costs = CostModel(perp_fees=PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP"))
        taker = costs.fee_micros(price_cc=PRICE_CC, qty_qc=QTY_QC, is_taker=True)
        maker = costs.fee_micros(price_cc=PRICE_CC, qty_qc=QTY_QC, is_taker=False)
        assert maker * 2 == taker

    def test_charging_makers_as_takers_still_overrides(self):
        """The conservative switch has to keep working on the perp path too."""
        costs = CostModel(
            perp_fees=PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP"),
            charge_maker_as_taker=True,
        )
        assert costs.fee_micros(
            price_cc=PRICE_CC, qty_qc=QTY_QC, is_taker=False
        ) == costs.fee_micros(price_cc=PRICE_CC, qty_qc=QTY_QC, is_taker=True)

    def test_the_schedule_is_recorded_on_the_result(self):
        """A number may never be quoted without the cost assumptions behind it."""
        recorded = CostModel(
            perp_fees=PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP")
        ).to_dict()["perp_fees"]
        assert recorded == {
            "maker_rate": "0.0002", "taker_rate": "0.0004", "exact": True
        }
        assert CostModel().to_dict()["perp_fees"] is None


class TestAPerpetualIsNeverPricedThroughTheEventFormula:
    """The bug this guards was not an overcharge, it was a sign flip.

    ``perp_costing`` took an event ``FeeSchedule`` and every perpetual strategy
    on the platform -- trend, funding carry, basis -- computed expected value
    through it. The quadratic is ``m * C * P * (1 - P)``, which assumes a price
    between zero and a dollar. A perpetual trades far above one: at $7.18 the
    term ``P * (1 - P)`` is -44.4, so the "fee" came back *negative* and every
    perp proposal was handed a rebate of about ninety dollars per twenty
    contracts, with its expected value inflated to match.

    A strategy cannot be measured, rejected or trusted while its costs carry the
    wrong sign, so this is a regression test on the sign before it is one on the
    magnitude.
    """

    def test_the_event_formula_really_does_go_negative_up_here(self):
        """The premise, asserted rather than asserted-about."""
        assert estimate_fee_micros(price_cc=71_800, qty_qc=2_000).fee_micros < 0

    def test_a_perpetual_position_is_charged_a_positive_fee(self):
        from kat.strategies.common import perp_costing

        costing = perp_costing(
            is_long=True, entry_cc=71_549, fair_cc=71_800, qty_qc=2_000,
            schedule=PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP"),
        )
        assert costing.fees_micros > 0

    def test_the_fee_is_both_legs_at_the_taker_rate(self):
        """A proposal that crosses to get in will cross to get out."""
        from kat.strategies.common import perp_costing

        schedule = PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP")
        costing = perp_costing(
            is_long=True, entry_cc=71_549, fair_cc=71_800, qty_qc=2_000,
            schedule=schedule,
        )
        assert costing.fees_micros == (
            perp_fee_micros(price_cc=71_549, qty_qc=2_000, schedule=schedule)
            + perp_fee_micros(price_cc=71_800, qty_qc=2_000, schedule=schedule)
        )

    def test_the_rebate_would_have_inflated_expected_value(self):
        """Why it mattered: a negative fee is added to the edge, not taken from
        it, so the worse the arithmetic the better the trade looked."""
        from kat.strategies.common import perp_costing

        costing = perp_costing(
            is_long=True, entry_cc=71_549, fair_cc=71_800, qty_qc=2_000,
            schedule=PerpFeeSchedule.from_fee_tiers(TIERS, "KXZECPERP"),
        )
        assert costing.net_micros < costing.gross_micros
