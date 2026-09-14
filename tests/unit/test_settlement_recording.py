"""Outcomes have to be written down or no settlement model can ever be trained.

`MispricingStrategy` converts a probability into a fair value, which is only
meaningful if the number is the chance the contract resolves YES. That target is
`SettlementLabel`, it needs an outcome per market, and nothing had ever written a
row to `settlements`. So the platform trained the target it *could* label -- a
50-centicent move in half an hour -- which cannot be read as a fair value.

`GET /markets?status=settled` looks like the answer and is not: thirty pages of
it returned six thousand provisional shards this platform never priced, while the
markets it *had* priced report `finalized` and were absent entirely.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kat.engine.settlements import SettlementRecorder


class _Market:
    def __init__(self, ticker, result, status="finalized"):
        self.ticker = ticker
        self.result = result
        self.status = status
        self.close_time = datetime(2026, 8, 30, tzinfo=UTC)
        self.event_ticker = "E"
        self.last_price_cc = 9_900


class _Rest:
    def __init__(self, results):
        self.results = results
        self.asked: list[str] = []

    async def get_market(self, ticker):
        self.asked.append(ticker)
        if ticker not in self.results:
            raise RuntimeError("no such market")
        return _Market(ticker, self.results[ticker])


class _Writer:
    def __init__(self):
        self.rows = []

    def enqueue(self, row, *, critical=False):
        self.rows.append(row)
        return True


def _recorder(results, pending, **kw):
    rec = SettlementRecorder(_Rest(results), writer=_Writer(), database=None, **kw)
    # Mirrors the real _pending, which subtracts what has already been settled.
    # Stubbing that away would test the stub instead of the dedup.
    rec._pending = lambda: _async(  # type: ignore[method-assign]
        [t for t in pending if t not in rec._settled]
    )
    return rec


async def _async(value):
    return value


class TestOutcomesAreRecorded:
    async def test_a_resolved_market_is_written(self):
        rec = _recorder({"A": "yes", "B": "no"}, ["A", "B"])
        assert await rec.poll_once() == 2
        assert {r.ticker: r.result for r in rec.writer.rows} == {"A": "yes", "B": "no"}

    async def test_an_unresolved_market_is_left_for_next_time(self):
        rec = _recorder({"A": ""}, ["A"])
        assert await rec.poll_once() == 0
        assert rec.writer.rows == []
        assert rec.unresolved == 1

    async def test_a_void_result_is_not_recorded_as_a_no(self):
        """A market that voided did not resolve NO, and teaching that would be a
        fact that never happened."""
        rec = _recorder({"A": "void"}, ["A"])
        assert await rec.poll_once() == 0
        assert rec.writer.rows == []

    async def test_a_failed_lookup_does_not_stop_the_batch(self):
        """B does not exist; A and C must still be recorded."""
        rec = _recorder({"A": "yes", "C": "no"}, ["A", "B", "C"])
        assert await rec.poll_once() == 2
        assert rec.rest.asked == ["A", "B", "C"]

    async def test_the_batch_bounds_the_requests(self):
        rec = _recorder(dict.fromkeys("ABCDE", "yes"), list("ABCDE"), batch=2)
        assert await rec.poll_once() == 2
        assert len(rec.rest.asked) == 2

    async def test_a_recorded_market_is_not_asked_about_again(self):
        """A resolved market stays resolved; asking again is wasted budget."""
        rec = _recorder({"A": "yes"}, ["A"])
        assert await rec.poll_once() == 1

        assert await rec.poll_once() == 0
        assert rec.rest.asked == ["A"], "the second pass must not re-query it"

    async def test_nothing_pending_costs_no_requests(self):
        rec = _recorder({}, [])
        assert await rec.poll_once() == 0
        assert rec.rest.asked == []


class TestTheBacklogIsWalkedNotItsHead:
    """Every pass used to take the same alphabetical head of the backlog.

    A market still open is retried rather than settled, so once the first batch
    were long-dated, every pass asked about those and nothing after them was ever
    reached: review reproduced 11,520 lookups of 120 markets in a day with 500
    resolved markets never asked about, and live the still_pending count only rose.
    """

    async def test_resolved_markets_behind_an_open_head_are_reached(self):
        still_open = {f"A{i:03d}": "" for i in range(4)}
        resolved = {f"B{i:03d}": "yes" for i in range(6)}
        rec = _recorder({**still_open, **resolved}, sorted(still_open) + sorted(resolved), batch=4)

        for _ in range(3):
            await rec.poll_once()

        assert rec._settled == set(resolved)

    async def test_the_walk_wraps_back_to_the_start(self):
        rec = _recorder(dict.fromkeys("ABCDE", ""), list("ABCDE"), batch=2)
        for _ in range(3):
            await rec.poll_once()

        assert rec.rest.asked == ["A", "B", "C", "D", "E", "A"]
