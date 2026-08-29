"""Requesting production data without production credentials must be loud.

Kalshi does not serve quotes to an unauthenticated client, but it does answer
200. Twelve thousand markets come back with no bid, no ask, no volume and an
empty orderbook, so nothing raises and nothing retries -- the platform just
watches a universe where nothing is ever tradeable, which looks exactly like a
quiet market. This is the one misconfiguration that produces no error at all.
"""

from __future__ import annotations

from kat.config.settings import AppSettings, TradingEnvironment


def _settings(**kalshi: object) -> AppSettings:
    base = {"read_production_market_data": False}
    base.update(kalshi)
    return AppSettings(environment=TradingEnvironment.PAPER, kalshi=base)


def test_paper_defaults_to_the_demo_exchange() -> None:
    assert _settings().kalshi.reads_production is False
    assert "demo" in _settings().kalshi.rest_base_url


def test_paper_can_read_production_data() -> None:
    """The whole point: real markets, no orders."""
    settings = _settings(read_production_market_data=True)
    assert settings.kalshi.reads_production is True
    assert "demo" not in settings.kalshi.rest_base_url


def test_reading_production_never_permits_an_order() -> None:
    """Two independent conditions, and this flag is neither of them."""
    settings = _settings(read_production_market_data=True)
    assert settings.environment is TradingEnvironment.PAPER
    assert settings.can_submit_real_orders is False


def test_the_flag_is_ignored_outside_paper() -> None:
    """In DEMO the platform really does send orders, so it must price them from
    the same exchange it sends them to."""
    settings = AppSettings(
        environment=TradingEnvironment.DEMO,
        kalshi={"read_production_market_data": True},
    )
    assert settings.kalshi.reads_production is False
    assert "demo" in settings.kalshi.rest_base_url


def test_production_uses_the_production_credential_slot() -> None:
    """Kalshi does not share credentials between the two exchanges, so the
    identity follows the exchange rather than the environment."""
    settings = _settings(read_production_market_data=True)
    assert settings.kalshi.active_credentials is settings.kalshi.prod_credentials

    demo = _settings()
    assert demo.kalshi.active_credentials is demo.kalshi.event_credentials
