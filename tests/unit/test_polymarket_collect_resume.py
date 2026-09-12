"""The collector's follow set survives a crash.

``collect()`` kept the listing and the follow set as locals, and ``main()``
re-enters ``collect()`` after a crash -- so every crash threw the follow set
away and forced a full sweep of about 4,000 markets, 25 to 30 minutes at the
polite rate, before following resumed. Each crash turned a blip into half an
hour of coarse sampling of exactly the markets chosen to be sampled finely.

No network and no database: the REST client and the connection are fakes that
record what they were asked for, which is the thing under test.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from kat.marketdata.book import OrderBook

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "polymarket_collect.py"

SLUGS = ["mkt-a", "mkt-b", "mkt-c", "mkt-d"]

#: Every fake book is identical, so ranking falls through to the slug
#: tie-break and the follow set is the first two slugs.
FOLLOW = 2
FOLLOWED = ["mkt-a", "mkt-b"]


@pytest.fixture(scope="module")
def collector() -> Iterator[ModuleType]:
    """The script, loaded by path.

    ``scripts/`` is not a package, and importing it through the working
    directory would make these pass or fail according to how pytest was
    launched.
    """
    spec = importlib.util.spec_from_file_location("polymarket_collect_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before executing: dataclasses resolves string annotations
    # through sys.modules, and fails on a module that is not in it.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class FakeREST:
    def __init__(
        self, *, crash_on_book: int | None = None, crash: Exception | None = None
    ) -> None:
        self.listings = 0
        self.books: list[str] = []
        self.crash_on_book = crash_on_book
        self.crash = crash or RuntimeError("unparseable book")

    async def get_open_markets(self, *, max_pages: int = 10) -> list[Any]:
        self.listings += 1
        return [SimpleNamespace(slug=slug, is_tradeable=True) for slug in SLUGS]

    async def get_book(self, slug: str) -> OrderBook:
        self.books.append(slug)
        if len(self.books) == self.crash_on_book:
            raise self.crash
        # A one-cent spread with a queue on both sides: a positive maker edge.
        return OrderBook.from_two_sided_levels(
            slug, bid_levels=[(4_000, 500)], ask_levels=[(4_100, 500)]
        )

    async def get_flow(self, slug: str) -> tuple[int, int]:
        # Non-zero: rank_tradeable only follows markets that have traded.
        return 12_000, 4_050


class FakeConn:
    def __init__(self) -> None:
        self.batches = 0

    async def executemany(self, sql: str, rows: list[Any]) -> None:
        self.batches += 1


class FakeDb:
    def __init__(self) -> None:
        self.conn = FakeConn()
        self.reconnects = 0

    async def reconnect(self) -> bool:
        self.reconnects += 1
        return True

    async def close(self) -> None:
        return None


class RecordingLog:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        def record(event: str, **fields: Any) -> None:
            self.records.append((level, event, fields))

        return record


@pytest.fixture
def log(collector: ModuleType, monkeypatch: pytest.MonkeyPatch) -> RecordingLog:
    # Pacing and backoff are politeness towards a real endpoint; against a fake
    # they only make the test slow.
    monkeypatch.setattr(collector, "RATE_LIMIT_PER_S", 1e9)
    monkeypatch.setattr(collector, "BACKOFF_S", 0.0)
    recorder = RecordingLog()
    monkeypatch.setattr(collector, "log", recorder)
    return recorder


async def _release_waiters(stop: asyncio.Event) -> None:
    """Let the backoff's ``stop.wait()`` tasks finish before the loop closes."""
    stop.set()
    await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("crash", "error"),
    [
        (RuntimeError("unparseable book for mkt-b"), "unparseable book for mkt-b"),
        (ConnectionResetError(), "ConnectionResetError"),
    ],
    ids=["with-message", "empty-message"],
)
async def test_a_crash_while_following_resumes_following_without_a_sweep(
    collector, log, crash, error
):
    # The sweep polls every market; the crash is on the second followed one.
    rest = FakeREST(crash_on_book=len(SLUGS) + 2, crash=crash)
    db = FakeDb()
    stop = asyncio.Event()

    code = await collector.supervise(
        db, rest, stop=stop, interval_s=0.0, max_cycles=2, follow=FOLLOW
    )
    await _release_waiters(stop)

    assert code == 0
    sweep, crashed_pass, after = (
        rest.books[: len(SLUGS)],
        rest.books[len(SLUGS) : len(SLUGS) + 2],
        rest.books[len(SLUGS) + 2 :],
    )
    assert sweep == SLUGS
    assert crashed_pass == FOLLOWED
    # The next attempt's first pass -- and every pass after it -- polls only
    # what was already chosen, and the board is never listed again.
    assert after == FOLLOWED * 2
    assert rest.listings == 1
    assert db.reconnects == 1

    [crashed] = [
        (level, fields)
        for level, event, fields in log.records
        if event == "polymarket.collect_crashed"
    ]
    level, fields = crashed
    assert level == "exception"
    assert fields["attempt"] == 1
    # Explicit, so the cause is in the line whether or not a traceback renders.
    assert fields["error"] == error
    assert fields["error_type"] == type(crash).__name__


async def test_the_six_hourly_refresh_still_relists_and_resweeps(collector, log):
    rest = FakeREST()
    db = FakeDb()
    stop = asyncio.Event()
    state = collector.FollowState()

    await collector.collect(
        db, rest, interval_s=0.0, stop=stop, max_cycles=2, follow=FOLLOW, state=state
    )
    assert rest.listings == 1
    assert state.quotable == set(FOLLOWED)
    assert rest.books == [*SLUGS, *FOLLOWED]

    # Six hours and a second later, surviving state must not mean a stale board.
    state.listed_at -= collector.REFRESH_MARKETS_S + 1
    await collector.collect(
        db, rest, interval_s=0.0, stop=stop, max_cycles=1, follow=FOLLOW, state=state
    )
    assert rest.listings == 2
    assert rest.books[len(SLUGS) + len(FOLLOWED) :] == SLUGS
    assert state.quotable == set(FOLLOWED)


async def test_a_crash_mid_sweep_sweeps_again_rather_than_follow_a_partial_ranking(
    collector, log
):
    """A ranking of the first market on the board is not a follow set.

    Only a completed sweep may choose what to follow; a crash partway through
    one leaves nothing chosen, so the next attempt starts the sweep over.
    """
    rest = FakeREST(crash_on_book=2)
    db = FakeDb()
    stop = asyncio.Event()

    code = await collector.supervise(
        db, rest, stop=stop, interval_s=0.0, max_cycles=1, follow=FOLLOW
    )
    await _release_waiters(stop)

    assert code == 0
    assert rest.books == ["mkt-a", "mkt-b", *SLUGS]
