"""Fee terms reach the trading path, and a failure to fetch them is survivable.

``FeeSchedule.from_series`` existed, ``get_series`` existed, and the
``fee_schedules`` mappings on the trading loop, execution engine and arbitrage
scanner existed -- and nothing connected them. The mapping stayed empty for the
life of every run, so every fee estimate in the platform fell back to an unknown
schedule.

That fallback is not neutral. It marks the estimate inexact, which inflates it
by 50% so an unguessable fee cannot make a marginal trade look profitable.
Correct when the terms are genuinely unknown; permanently wrong when they are
one request away. Every strategy costed its round trip half again over the odds
and declined trades on an edge it did have.
"""

from __future__ import annotations

import pytest

# The loader caches per series for the life of the process, which is correct in
# production and would leak between tests.
from kat.engine import ingestion as ingestion_module
from kat.engine.ingestion import load_fee_schedules
from kat.kalshi.fees import BASE_COEFFICIENT, FeeSchedule


@pytest.fixture(autouse=True)
def clear_series_cache():
    ingestion_module._SERIES_SCHEDULES.clear()
    yield
    ingestion_module._SERIES_SCHEDULES.clear()


class _Rest:
    """Stands in for EventContractsREST.get_series."""

    def __init__(self, *, payload=None, fails: set[str] | None = None) -> None:
        self.payload = payload if payload is not None else {
            "fee_type": "quadratic",
            "fee_multiplier": 1,
        }
        self.fails = fails or set()
        self.calls: list[str] = []

    async def get_series(self, series_ticker: str):
        self.calls.append(series_ticker)
        if series_ticker in self.fails:
            raise RuntimeError("series lookup failed")
        return self.payload


def _definitions(*pairs: tuple[str, str]) -> dict[str, dict[str, object]]:
    return {ticker: {"series_ticker": series} for ticker, series in pairs}


class TestSchedulesReachTheTradingPath:
    async def test_every_watched_market_gets_its_series_terms(self):
        rest = _Rest()
        into: dict[str, FeeSchedule] = {}

        applied = await load_fee_schedules(
            rest,
            _definitions(
                ("KXPL-27-CFC", "KXPL"),
                ("KXPL-27-ARS", "KXPL"),
                ("KXMARS-99", "KXMARS"),
            ),
            into=into,
        )

        assert applied == 3
        assert set(into) == {"KXPL-27-CFC", "KXPL-27-ARS", "KXMARS-99"}
        assert all(s.exact for s in into.values())
        assert into["KXPL-27-CFC"].multiplier == BASE_COEFFICIENT

    async def test_a_series_is_fetched_once_however_many_markets_share_it(self):
        """Sixty watched markets come from a handful of series."""
        rest = _Rest()
        into: dict[str, FeeSchedule] = {}

        await load_fee_schedules(
            rest,
            _definitions(*[(f"KXPL-27-{i}", "KXPL") for i in range(20)]),
            into=into,
        )

        assert rest.calls == ["KXPL"]
        assert len(into) == 20

    async def test_the_cache_survives_a_watchlist_refresh(self):
        rest = _Rest()
        into: dict[str, FeeSchedule] = {}

        await load_fee_schedules(rest, _definitions(("A-1", "SA")), into=into)
        await load_fee_schedules(rest, _definitions(("A-2", "SA")), into=into)

        assert rest.calls == ["SA"]

    async def test_markets_that_left_the_watchlist_are_dropped(self):
        """The mapping is read by ticker on the trading path, so it cannot grow."""
        rest = _Rest()
        into: dict[str, FeeSchedule] = {}

        await load_fee_schedules(
            rest, _definitions(("A-1", "SA"), ("A-2", "SA")), into=into
        )
        assert len(into) == 2

        await load_fee_schedules(rest, _definitions(("A-2", "SA")), into=into)
        assert set(into) == {"A-2"}


class TestAFailedLookupIsSurvivable:
    async def test_a_failing_series_leaves_its_markets_on_the_fallback(self):
        rest = _Rest(fails={"BAD"})
        into: dict[str, FeeSchedule] = {}

        applied = await load_fee_schedules(
            rest, _definitions(("GOOD-1", "GOOD"), ("BAD-1", "BAD")), into=into
        )

        assert applied == 1
        assert "GOOD-1" in into
        assert "BAD-1" not in into

    async def test_a_market_with_no_series_is_skipped_not_guessed(self):
        """The series is not derived from the ticker string; absent means absent."""
        rest = _Rest()
        into: dict[str, FeeSchedule] = {}

        applied = await load_fee_schedules(
            rest, {"KXPL-27-CFC": {"series_ticker": ""}}, into=into
        )

        assert applied == 0
        assert into == {}
        assert rest.calls == []

    async def test_an_unreadable_payload_is_kept_inexact(self):
        """A schedule that cannot be read must keep its safety margin."""
        rest = _Rest(payload={"fee_type": "quadratic", "fee_multiplier": None})
        into: dict[str, FeeSchedule] = {}

        await load_fee_schedules(rest, _definitions(("A-1", "SA")), into=into)

        assert into["A-1"].multiplier == BASE_COEFFICIENT
        assert not into["A-1"].exact

    async def test_nothing_to_do_costs_no_requests(self):
        rest = _Rest()
        into: dict[str, FeeSchedule] = {}

        assert await load_fee_schedules(rest, {}, into=into) == 0
        assert rest.calls == []
