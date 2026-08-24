"""Shared pytest fixtures.

The test suite runs with no external services by default: SQLite stands in for
PostgreSQL and nothing touches the network. Tests that genuinely need a live
database or the Kalshi API are marked ``integration`` / ``network`` and are
skipped unless the corresponding environment variable is set, so a clean
checkout can run the full default suite immediately.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "packages") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packages"))

from kat.common.enums import MarketType, TradingEnvironment  # noqa: E402
from kat.common.timeutil import SimulatedClock  # noqa: E402
from kat.config.settings import EventRiskSettings, PerpRiskSettings  # noqa: E402
from kat.marketdata.book import OrderBook  # noqa: E402
from kat.marketdata.state import MarketState  # noqa: E402
from kat.risk.event_risk import EventAccountState, EventRiskEngine  # noqa: E402
from kat.risk.perp_risk import PerpAccountState, PerpRiskEngine  # noqa: E402

# A fixed simulated "now" so tests never depend on wall-clock time.
NOW_MS = 1_700_000_000_000


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW_MS)


@pytest.fixture
def now_ms() -> int:
    return NOW_MS


@pytest.fixture
def liquid_event_book() -> OrderBook:
    """A tight, deep event-contract book that passes every quality gate.

    YES bids at $0.40/$0.39; NO bids at $0.59/$0.58, which reflect to YES asks at
    $0.41/$0.42. Spread is one cent.
    """
    return OrderBook.from_event_contract_levels(
        "TEST-EVENT-MKT",
        yes_levels=[(4000, 50_000), (3900, 50_000)],
        no_levels=[(5900, 50_000), (5800, 50_000)],
    )


@pytest.fixture
def liquid_perp_book() -> OrderBook:
    return OrderBook.from_two_sided_levels(
        "TEST-PERP",
        bid_levels=[(4990, 50_000), (4980, 50_000)],
        ask_levels=[(5010, 50_000), (5020, 50_000)],
    )


@pytest.fixture
def event_state(now_ms: int) -> EventAccountState:
    """A $200 event-contract account -- the small-account case that must work."""
    state = EventAccountState(balance_micros=200_000_000, updated_at_ms=now_ms)
    state.peak_equity_micros = state.equity_micros
    return state


@pytest.fixture
def perp_state(now_ms: int) -> PerpAccountState:
    """A $200 perpetual account with known equity."""
    return PerpAccountState(
        equity_micros=200_000_000,
        equity_source="test",
        equity_is_known=True,
        updated_at_ms=now_ms,
    )


@pytest.fixture
def event_risk() -> EventRiskEngine:
    return EventRiskEngine(EventRiskSettings())


@pytest.fixture
def perp_risk() -> PerpRiskEngine:
    """Perp engine with configured budgets wide open.

    The configured exposure fraction is set to 1.00 so that the 1.00x invariant
    is the binding constraint in tests, rather than a softer configured limit
    masking whether the invariant itself works.
    """
    return PerpRiskEngine(
        PerpRiskSettings(
            max_exposure_pct_equity=Decimal("1.0"),
            max_position_pct_equity=Decimal("1.0"),
            max_order_notional_usd=Decimal("1000000"),
        )
    )


@pytest.fixture
def event_market_state() -> MarketState:
    return MarketState(MarketType.EVENT)


@pytest.fixture
def perp_market_state() -> MarketState:
    return MarketState(MarketType.PERP)


@pytest.fixture
async def sqlite_db(tmp_path: Path) -> AsyncIterator[object]:
    """A real database on SQLite, with the full schema created.

    Real SQL against a real engine -- not a mock. The same models run on
    PostgreSQL in production.
    """
    from kat.config.settings import DatabaseSettings
    from kat.db.engine import Database

    db = Database(DatabaseSettings(url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"))
    await db.connect()
    await db.create_all()
    try:
        yield db
    finally:
        await db.disconnect()


@pytest.fixture
def paper_environment() -> TradingEnvironment:
    return TradingEnvironment.PAPER


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires a live database")
    config.addinivalue_line("markers", "network: requires network access to Kalshi")
    config.addinivalue_line("markers", "slow: long running")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip network tests unless explicitly enabled.

    Network tests hit the real Kalshi API. They are valuable and they are real,
    but a test suite that fails because an exchange is down is a test suite
    people learn to ignore.
    """
    if os.environ.get("KAT_TEST_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="set KAT_TEST_NETWORK=1 to run tests that hit the Kalshi API")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:
    """Keep test output readable without disabling logging entirely."""
    import logging

    from kat.observability.logging import configure_logging

    configure_logging("ERROR")
    logging.getLogger().setLevel(logging.ERROR)
    yield
