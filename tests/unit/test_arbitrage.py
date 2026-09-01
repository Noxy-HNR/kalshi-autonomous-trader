"""Arbitrage detection and pricing.

Two properties are tested throughout:

1. A **real** inconsistency is found, with the correct legs.
2. An inconsistency that is real but too small to survive fees, depth or leg
   risk is **declined** -- because most apparent prediction-market arbitrage is
   an artefact of quoting the midpoint, and a scanner that cannot tell the
   difference will lose money enthusiastically.
"""

from __future__ import annotations

import pytest

from kat.arbitrage.logical import (
    LadderRung,
    find_complementary_pair,
    find_ladder_violation,
    find_mutually_exclusive_set,
)
from kat.arbitrage.scanner import opportunity_to_intents
from kat.common.enums import OutcomeSide
from kat.kalshi.fees import FeeSchedule
from kat.marketdata.book import OrderBook

SCHEDULE = FeeSchedule.from_series("quadratic", 0.07)
DEEP = 5_000_000


def book(ticker: str, yes_bid_cc: int, no_bid_cc: int, depth: int = DEEP) -> OrderBook:
    """A two-sided event book.

    Kalshi publishes two *bid* arrays, so the YES ask is derived: a NO bid at
    ``p`` is a YES ask at ``1 - p``.
    """
    return OrderBook.from_event_contract_levels(
        ticker,
        yes_levels=[(yes_bid_cc, depth), (yes_bid_cc - 100, depth)],
        no_levels=[(no_bid_cc, depth), (no_bid_cc - 100, depth)],
    )


class TestComplementaryPair:
    def test_finds_a_genuine_underpricing(self):
        """Both sides buyable for $0.20 against a $1.00 settlement."""
        opportunity = find_complementary_pair(
            "MKT", book("MKT", 9000, 9000), qty_qc=500, fee_schedule=SCHEDULE
        )
        assert opportunity is not None
        assert opportunity.is_actionable
        assert opportunity.gross_edge_micros > 0
        assert opportunity.net_edge_micros > 0
        assert {leg.side for leg in opportunity.legs} == {OutcomeSide.YES, OutcomeSide.NO}

    def test_declines_when_the_pair_costs_more_than_a_dollar(self):
        opportunity = find_complementary_pair(
            "MKT", book("MKT", 4000, 5500), qty_qc=500, fee_schedule=SCHEDULE
        )
        assert opportunity is not None
        assert not opportunity.is_actionable
        assert opportunity.reason == "no gross edge"

    def test_declines_a_two_cent_edge_that_fees_consume(self):
        """The trap: $0.98 for a $1.00 payout looks free and is not.

        Two legs near $0.50 cost about 3.5 cents in round-trip fees, so a
        two-cent gross edge is a loss.
        """
        opportunity = find_complementary_pair(
            "MKT", book("MKT", 5100, 5100), qty_qc=500, fee_schedule=SCHEDULE
        )
        assert opportunity is not None
        assert opportunity.gross_edge_micros > 0, "the gross edge is real"
        assert opportunity.net_edge_micros < 0, "and fees turn it into a loss"
        assert not opportunity.is_actionable
        assert opportunity.reason == "fees consume the gross edge"

    def test_returns_none_when_a_side_is_empty(self):
        one_sided = OrderBook.from_event_contract_levels(
            "MKT", yes_levels=[(4000, DEEP)], no_levels=[]
        )
        assert find_complementary_pair("MKT", one_sided, fee_schedule=SCHEDULE) is None


class TestMutuallyExclusiveSet:
    @pytest.fixture
    def cheap_set(self) -> dict[str, OrderBook]:
        # YES ask of each is 1.00 - 0.70 = 0.30; three of them cost $0.90.
        return {f"M{i}": book(f"M{i}", 2500, 7000) for i in range(3)}

    def test_finds_a_set_priced_below_one_dollar(self, cheap_set):
        opportunity = find_mutually_exclusive_set(
            "EVT",
            cheap_set,
            qty_qc=500,
            fee_schedules=dict.fromkeys(cheap_set, SCHEDULE),
            is_mutually_exclusive=True,
            is_exhaustive=True,
        )
        assert opportunity is not None
        assert opportunity.is_actionable
        assert len(opportunity.legs) == 3

    def test_refuses_a_set_that_is_not_mutually_exclusive(self, cheap_set):
        """Two outcomes could both resolve YES, so the payout is not $1.00."""
        assert (
            find_mutually_exclusive_set(
                "EVT", cheap_set, is_mutually_exclusive=False, is_exhaustive=True
            )
            is None
        )

    def test_refuses_a_set_that_is_not_exhaustive(self, cheap_set):
        """An unlisted outcome means buying everything can pay nothing at all."""
        assert (
            find_mutually_exclusive_set(
                "EVT", cheap_set, is_mutually_exclusive=True, is_exhaustive=False
            )
            is None
        )

    def test_declines_a_set_priced_at_or_above_a_dollar(self):
        # YES ask 1.00 - 0.65 = 0.35 each; three cost $1.05.
        expensive = {f"M{i}": book(f"M{i}", 3000, 6500) for i in range(3)}
        opportunity = find_mutually_exclusive_set(
            "EVT",
            expensive,
            fee_schedules=dict.fromkeys(expensive, SCHEDULE),
            is_mutually_exclusive=True,
            is_exhaustive=True,
        )
        assert opportunity is not None
        assert not opportunity.is_actionable


class TestLadderMonotonicity:
    """The direction of this trade is the whole point.

    Buying the **looser YES** and the **stricter NO** pays $1.00 in the outer
    bands and $2.00 in the middle band, so it can never lose. The mirror trade
    pays *nothing* in the middle band and is a directional bet.
    """

    def test_a_consistent_ladder_yields_nothing(self):
        # P(>50) = 0.60, P(>60) = 0.40: correctly ordered.
        rungs = [
            LadderRung("T50", 50.0, book("T50", 5900, 4000), SCHEDULE),
            LadderRung("T60", 60.0, book("T60", 3900, 6000), SCHEDULE),
        ]
        assert find_ladder_violation(rungs, qty_qc=500) is None

    def test_finds_an_inverted_ladder(self):
        # P(>50) = 0.30 while P(>60) = 0.70, which is impossible.
        rungs = [
            LadderRung("T50", 50.0, book("T50", 2900, 7000), SCHEDULE),
            LadderRung("T60", 60.0, book("T60", 6900, 3000), SCHEDULE),
        ]
        opportunity = find_ladder_violation(rungs, qty_qc=500)
        assert opportunity is not None
        assert opportunity.is_actionable

    def test_buys_the_looser_yes_and_the_stricter_no(self):
        """Getting this backwards produces a trade that pays nothing in the
        middle band, so the sides are asserted explicitly."""
        rungs = [
            LadderRung("T50", 50.0, book("T50", 2900, 7000), SCHEDULE),
            LadderRung("T60", 60.0, book("T60", 6900, 3000), SCHEDULE),
        ]
        opportunity = find_ladder_violation(rungs, qty_qc=500)
        assert opportunity is not None
        sides = {leg.ticker: leg.side for leg in opportunity.legs}
        assert sides["T50"] is OutcomeSide.YES, "the looser threshold is bought as YES"
        assert sides["T60"] is OutcomeSide.NO, "the stricter threshold is bought as NO"
        assert opportunity.detail["buy_yes_ticker"] == "T50"
        assert opportunity.detail["buy_no_ticker"] == "T60"

    def test_declines_a_one_cent_violation(self):
        rungs = [
            LadderRung("T50", 50.0, book("T50", 4900, 5100), SCHEDULE),
            LadderRung("T60", 60.0, book("T60", 5100, 4900), SCHEDULE),
        ]
        opportunity = find_ladder_violation(rungs, qty_qc=500)
        assert opportunity is not None
        assert not opportunity.is_actionable

    def test_a_single_rung_is_not_a_ladder(self):
        assert find_ladder_violation([LadderRung("T", 1.0, book("T", 4000, 5000))]) is None


class TestDepthAndLegRisk:
    def test_a_thin_book_reduces_size_and_confidence(self):
        """An arbitrage that needs the whole displayed book will not fill.

        ``book()`` lays two levels of ``depth`` on each side, so 200 per level is
        400 available in total; the request for 5000 must be cut to that, and
        needing all of it must reduce the fill probability below one.
        """
        thin = find_complementary_pair(
            "THIN", book("THIN", 9000, 9000, depth=200), qty_qc=5000, fee_schedule=SCHEDULE
        )
        assert thin is not None
        assert thin.qty_qc <= 400, "size must be capped by available depth"
        assert thin.qty_qc < 5000, "the requested size was not available"
        assert thin.execution_probability < 1

    def test_execution_probability_is_one_on_a_deep_book(self):
        deep = find_complementary_pair(
            "DEEP", book("DEEP", 9000, 9000), qty_qc=500, fee_schedule=SCHEDULE
        )
        assert deep is not None
        assert deep.execution_probability == 1

    def test_worst_case_is_recorded_and_is_a_loss(self):
        """If the hardest leg fails we hold the rest unhedged; that is priced."""
        opportunity = find_complementary_pair(
            "MKT", book("MKT", 9000, 9000), qty_qc=500, fee_schedule=SCHEDULE
        )
        assert opportunity is not None
        assert opportunity.worst_case_micros < opportunity.net_edge_micros
        assert opportunity.residual_exposure_micros > 0


class TestIntentConstruction:
    def test_declined_opportunities_produce_no_intents(self):
        opportunity = find_complementary_pair(
            "MKT", book("MKT", 4000, 5500), qty_qc=500, fee_schedule=SCHEDULE
        )
        assert opportunity is not None
        assert opportunity_to_intents(opportunity) == []

    def test_intents_are_ordered_hardest_leg_first(self):
        """Failing on the first leg costs nothing; failing on the last leaves an
        unhedged position entered because we believed we were hedged."""
        lopsided = OrderBook.from_event_contract_levels(
            "MKT",
            # YES bid side is deep, NO bid side (which sets the YES ask) is thin.
            yes_levels=[(9000, DEEP)],
            no_levels=[(9000, 400)],
        )
        opportunity = find_complementary_pair(
            "MKT", lopsided, qty_qc=300, fee_schedule=SCHEDULE
        )
        assert opportunity is not None and opportunity.is_actionable
        intents = opportunity_to_intents(opportunity)
        assert len(intents) == 2
        assert intents[0].metadata["leg_index"] == 0
        assert intents[0].metadata["hardest_leg_first"] is True
        # The thin side must come first.
        depths = [leg.available_qc for leg in sorted(opportunity.legs, key=lambda x: x.available_qc)]
        assert depths[0] <= depths[1]

    def test_all_legs_share_one_correlation_id(self):
        opportunity = find_complementary_pair(
            "MKT", book("MKT", 9000, 9000), qty_qc=500, fee_schedule=SCHEDULE
        )
        assert opportunity is not None
        intents = opportunity_to_intents(opportunity)
        assert len({i.correlation_id for i in intents}) == 1

    def test_expected_value_is_not_double_counted_across_legs(self):
        """The set is priced once; attributing its EV to every leg would make a
        two-leg trade look twice as good as it is."""
        opportunity = find_complementary_pair(
            "MKT", book("MKT", 9000, 9000), qty_qc=500, fee_schedule=SCHEDULE
        )
        assert opportunity is not None
        intents = opportunity_to_intents(opportunity)
        total = sum(i.expected_value_micros or 0 for i in intents)
        assert total == opportunity.expected_value_micros


# ---------------------------------------------------------------------------
# what the scanner keeps
# ---------------------------------------------------------------------------


class TestOnlyEvidenceIsRecorded:
    """The scanner persisted every pair it looked at, not every one that meant
    something.

    Measured on the live database: 901,781 rows in seven hours, 2,086 a minute,
    all ``actioned=False``, all with the reason "no gross edge" and a negative
    edge, covering the same 1,366 leg sets about 660 times each. Two prices that
    do not cross is what prices normally do; recording it says only that the
    scanner ran, which the counters already say.
    """

    @staticmethod
    def _candidate(*, gross: int, ev: int, qty_qc: int = 500, reason: str = ""):
        from decimal import Decimal

        from kat.arbitrage.pricing import PricedOpportunity

        return PricedOpportunity(
            kind="complementary_pair",
            legs=[],
            settlement_micros=1_000_000,
            gross_edge_micros=gross,
            fees_micros=0,
            slippage_micros=0,
            net_edge_micros=ev,
            execution_probability=Decimal("0.9"),
            residual_exposure_micros=0,
            worst_case_micros=0,
            expected_value_micros=ev,
            qty_qc=qty_qc,
            reason=reason,
        )

    @staticmethod
    def _scanner(**kwargs):
        from kat.arbitrage.scanner import ArbitrageScanner
        from kat.common.enums import MarketType
        from kat.common.timeutil import SimulatedClock
        from kat.marketdata.state import MarketState

        class _Writer:
            def __init__(self) -> None:
                self.rows: list[object] = []

            def enqueue(self, row: object, *, critical: bool = False) -> bool:
                self.rows.append(row)
                return True

        writer = _Writer()
        scanner = ArbitrageScanner(
            market_state=MarketState(MarketType.EVENT),
            clock=SimulatedClock(0),
            writer=writer,  # type: ignore[arg-type]
            **kwargs,
        )
        return scanner, writer

    def test_a_pair_that_does_not_cross_is_not_recorded(self):
        """The exact shape of all 901,781 rows."""
        scanner, writer = self._scanner()
        scanner._record(
            self._candidate(gross=0, ev=-17_100, reason="no gross edge")
        )
        assert writer.rows == []

    def test_a_near_miss_is_still_recorded(self):
        """A real cross that costs took away is the case record_declined is for."""
        scanner, writer = self._scanner()
        scanner._record(
            self._candidate(gross=50_000, ev=-2_000, reason="fees exceed the edge")
        )
        assert len(writer.rows) == 1
        assert writer.rows[0].actioned is False

    def test_an_actionable_opportunity_is_always_recorded(self):
        scanner, writer = self._scanner()
        scanner._record(self._candidate(gross=80_000, ev=40_000))
        assert len(writer.rows) == 1
        assert writer.rows[0].actioned is True

    def test_record_declined_off_still_keeps_the_actionable_one(self):
        scanner, writer = self._scanner(record_declined=False)
        scanner._record(self._candidate(gross=50_000, ev=-2_000))
        assert writer.rows == []
        scanner._record(self._candidate(gross=80_000, ev=40_000))
        assert len(writer.rows) == 1

    def test_a_flood_of_non_crossing_pairs_writes_nothing(self):
        """The regression, at the rate that produced the table."""
        scanner, writer = self._scanner()
        for _ in range(2_086):
            scanner._record(
                self._candidate(gross=-100, ev=-17_100, reason="no gross edge")
            )
        assert writer.rows == []


class TestExclusivityIsNotExhaustiveness:
    """The trap that sits where the arbitrage scanner is wired up.

    Mutually exclusive means *at most* one outcome resolves YES. Exhaustive
    means *at least* one does. Only both together make a set sum to a dollar at
    settlement, and Kalshi publishes only the first.

    Measured on the live board: 1,491 mutually exclusive sets, 36 of which trade
    below a dollar net of fees and offer up to 8,478 centicents on the dollar.
    None is an arbitrage. KXSTATE51-29 prices eight candidate states at 14.20
    cents in total because the market believes there will be no 51st state at
    all -- a bet at six to one against, wearing the costume of free money.
    """

    def test_the_scanner_requires_both_flags(self):
        from pathlib import Path

        import kat.arbitrage.scanner as scanner

        source = Path(scanner.__file__).read_text(encoding="utf-8")
        assert "if not (exclusive and exhaustive):" in source, (
            "the mutually-exclusive branch must require exhaustiveness, not "
            "exclusivity alone"
        )

    def test_flags_default_to_refusing(self):
        """An event nobody has vouched for is not assumed exhaustive."""
        from kat.arbitrage.scanner import ArbitrageScanner
        from kat.common.enums import MarketType
        from kat.common.timeutil import SimulatedClock
        from kat.marketdata.state import MarketState

        scanner = ArbitrageScanner(
            market_state=MarketState(MarketType.EVENT), clock=SimulatedClock(0)
        )
        assert scanner.event_flags.get("ANYTHING", (False, False)) == (False, False)
