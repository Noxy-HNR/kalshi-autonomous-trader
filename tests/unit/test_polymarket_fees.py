"""A rebate is a negative cost, and negative costs break arithmetic written for costs.

The platform measured eleven structures on Kalshi and every one lost, dominated
by a fee that peaks at 350 basis points and a minimum tick that is a third of the
notional on a three-cent contract. Polymarket US prices the same instrument on
the same quadratic form with a maker coefficient of **-0.0125**: it pays the
maker. That single sign is why the venue was ported, so the sign is what these
tests are about.

Every case below is one where a rebate could silently flow the wrong way and
make a trade look better than it is.
"""

from __future__ import annotations

from decimal import Decimal

from kat.kalshi.fees import estimate_fee_micros
from kat.polymarket.fees import (
    MAKER_THETA,
    TAKER_THETA,
    PolymarketFeeSchedule,
    polymarket_fee_micros,
    polymarket_round_trip_micros,
)

#: A book sampled live from the exchange: 15.4 bid / 16.0 ask, mid 15.7.
MID_CC = 1_570
HUNDRED = 10_000  # centi-contracts

EXACT = PolymarketFeeSchedule.from_market({"feeCoefficient": "0.06"})


class TestTheMakerIsPaid:
    def test_a_maker_fill_returns_a_negative_fee(self):
        fee = polymarket_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED, schedule=EXACT, is_taker=False
        )
        assert fee < 0

    def test_a_taker_fill_still_costs(self):
        assert polymarket_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED, schedule=EXACT, is_taker=True
        ) > 0

    def test_a_passive_round_trip_is_paid_for(self):
        """Both legs resting: the exchange pays for the privilege of quoting."""
        assert polymarket_round_trip_micros(
            entry_price_cc=1_540, exit_price_cc=1_600, qty_qc=HUNDRED,
            schedule=EXACT, entry_is_taker=False, exit_is_taker=False,
        ) < 0

    def test_the_schedule_says_so_without_arithmetic(self):
        assert EXACT.pays_makers is True

    def test_this_is_the_whole_reason_for_the_port(self):
        """The same fill, on both venues. Kalshi charges a maker the taker rate
        on a plain quadratic series; Polymarket pays."""
        kalshi = estimate_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED, is_taker=False
        ).fee_micros
        poly = polymarket_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED, schedule=EXACT, is_taker=False
        )
        assert kalshi > 0 > poly
        # Over a round trip the gap exceeds a dollar per hundred contracts.
        assert (kalshi - poly) * 2 > 1_000_000


class TestRoundingNeverFavoursUs:
    def test_a_cost_rounds_up(self):
        """Ceiling on a positive takes it away from zero."""
        tiny = PolymarketFeeSchedule(taker_theta=Decimal("0.06"), exact=True)
        # One centi-contract: a fee far below a whole centicent.
        assert polymarket_fee_micros(price_cc=5_000, qty_qc=1, schedule=tiny) > 0

    def test_a_rebate_rounds_toward_zero(self):
        """The trap. Rounding a rebate away from zero would pay us more than the
        exchange does, on every fill, forever."""
        schedule = PolymarketFeeSchedule(
            maker_theta=Decimal("-0.0125"), exact=True
        )
        fee = polymarket_fee_micros(
            price_cc=5_000, qty_qc=1, schedule=schedule, is_taker=False
        )
        # A hundredth of a contract earns less than one centicent, so the
        # conservative answer is zero rather than a fraction in our favour.
        assert fee == 0

    def test_the_rebate_is_never_larger_than_the_exact_arithmetic(self):
        exact = -(MAKER_THETA * Decimal(100) * Decimal("0.157") * Decimal("0.843"))
        got = -polymarket_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED, schedule=EXACT, is_taker=False
        )
        assert Decimal(got) / Decimal(1_000_000) <= exact


class TestTheSafetyMarginPointsTheRightWay:
    def test_an_unknown_cost_is_inflated(self):
        guessed = PolymarketFeeSchedule()          # exact=False
        known = PolymarketFeeSchedule(exact=True)
        assert polymarket_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED, schedule=guessed
        ) > polymarket_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED, schedule=known
        )

    def test_an_unknown_rebate_is_shrunk_not_grown(self):
        """Multiplying an uncertain rebate by the margin would make an unknown
        schedule look *more* profitable, which is a safety margin inverted."""
        guessed = polymarket_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED,
            schedule=PolymarketFeeSchedule(), is_taker=False,
        )
        known = polymarket_fee_micros(
            price_cc=MID_CC, qty_qc=HUNDRED,
            schedule=PolymarketFeeSchedule(exact=True), is_taker=False,
        )
        assert known < guessed <= 0        # less negative == smaller rebate


class TestItRefusesToGuess:
    def test_a_market_without_a_coefficient_is_inexact(self):
        assert PolymarketFeeSchedule.from_market({}).exact is False
        assert PolymarketFeeSchedule.from_market(None).exact is False

    def test_an_unparseable_coefficient_falls_back(self):
        assert PolymarketFeeSchedule.from_market(
            {"feeCoefficient": "nonsense"}
        ).exact is False

    def test_a_negative_taker_coefficient_is_refused(self):
        """That would mean the exchange pays both sides of every trade. It is a
        misreading of the payload, not a gift, and banking on it would size
        every position against income that does not exist."""
        s = PolymarketFeeSchedule.from_market({"feeCoefficient": "-0.06"})
        assert s.exact is False
        assert s.taker_theta == TAKER_THETA

    def test_the_published_coefficient_is_taken_when_given(self):
        s = PolymarketFeeSchedule.from_market({"feeCoefficient": "0.06"})
        assert s.exact is True
        assert s.taker_theta == Decimal("0.06")


class TestDegenerateInputs:
    def test_nothing_traded_costs_and_earns_nothing(self):
        assert polymarket_fee_micros(price_cc=MID_CC, qty_qc=0) == 0
        assert polymarket_fee_micros(price_cc=0, qty_qc=HUNDRED) == 0

    def test_a_settled_price_earns_no_rebate(self):
        """At zero or a dollar the quadratic term vanishes; quoting a resolved
        market must not look like free income."""
        for price in (0, 10_000, 10_001):
            assert polymarket_fee_micros(
                price_cc=price, qty_qc=HUNDRED, schedule=EXACT, is_taker=False
            ) == 0


class TestTheSimulatorChargesThisVenuesFee:
    """A backtest of this venue that charges Kalshi's fee measures nothing.

    The two share a formula and differ in its coefficients, so the wrong one
    does not raise -- it silently returns a plausible number with the wrong sign
    on the maker side, which is the only side that matters here.
    """

    def test_a_maker_fill_is_income_through_the_cost_model(self):
        from kat.simulation.costs import CostModel

        costs = CostModel(polymarket_fees=EXACT)
        assert costs.fee_micros(price_cc=MID_CC, qty_qc=HUNDRED, is_taker=False) < 0

    def test_a_taker_fill_is_still_a_cost(self):
        from kat.simulation.costs import CostModel

        costs = CostModel(polymarket_fees=EXACT)
        assert costs.fee_micros(price_cc=MID_CC, qty_qc=HUNDRED, is_taker=True) > 0

    def test_the_event_path_is_untouched_when_unset(self):
        from kat.simulation.costs import CostModel

        assert CostModel().fee_micros(
            price_cc=5_000, qty_qc=HUNDRED, is_taker=False
        ) > 0

    def test_charging_makers_as_takers_still_overrides(self):
        """The conservative switch must reach this path too, and on this venue
        it does something unusual: it turns income into a cost."""
        from kat.simulation.costs import CostModel

        costs = CostModel(polymarket_fees=EXACT, charge_maker_as_taker=True)
        assert costs.fee_micros(price_cc=MID_CC, qty_qc=HUNDRED, is_taker=False) > 0

    def test_the_schedule_is_recorded_on_the_result(self):
        from kat.simulation.costs import CostModel

        assert CostModel(polymarket_fees=EXACT).to_dict()["polymarket_fees"] == {
            "taker_theta": "0.06", "maker_theta": "-0.0125", "exact": True
        }
        assert CostModel().to_dict()["polymarket_fees"] is None
