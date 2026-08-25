"""Reading production data must never become sending production orders.

The demo exchange has too little liquidity to paper trade against, so the
platform can point its market-data feed at production while still in PAPER.
That is only acceptable if the flag cannot, in any combination, cause a real
order -- which is what these pin.
"""

from __future__ import annotations

import pytest

from kat.common.enums import TradingEnvironment
from kat.config.settings import AppSettings, KalshiCredentials, KalshiSettings

DEMO = "demo.kalshi.co"


def is_production(url: str) -> bool:
    """Production hosts differ per channel (external-api, external-api-ws, ...),
    so identify them by not being the demo host rather than by one prefix."""
    return "kalshi.com" in url and DEMO not in url


def kalshi(**kw) -> KalshiSettings:
    return KalshiSettings(**kw)


class TestExchangeSelection:
    def test_paper_reads_demo_by_default(self):
        k = kalshi(environment=TradingEnvironment.PAPER)
        assert k.reads_production is False
        assert DEMO in k.rest_base_url
        assert DEMO in k.ws_url

    def test_paper_can_read_production_when_asked(self):
        k = kalshi(environment=TradingEnvironment.PAPER, read_production_market_data=True)
        assert k.reads_production is True
        assert is_production(k.rest_base_url)
        assert is_production(k.ws_url)

    def test_live_always_reads_production(self):
        k = kalshi(environment=TradingEnvironment.LIVE)
        assert k.reads_production is True
        assert is_production(k.rest_base_url)

    def test_demo_ignores_the_flag(self):
        """DEMO really does send orders, so it must price from where it sends.

        Reading production while sending to demo would price against a book the
        orders never reach.
        """
        k = kalshi(environment=TradingEnvironment.DEMO, read_production_market_data=True)
        assert k.reads_production is False
        assert DEMO in k.rest_base_url

    def test_every_url_moves_together(self):
        """No mixed state where REST is production and the socket is demo."""
        k = kalshi(environment=TradingEnvironment.PAPER, read_production_market_data=True)
        for url in (k.rest_base_url, k.ws_url, k.perp_rest_base_url, k.perp_ws_url):
            assert is_production(url), url


class TestCredentialSelection:
    def test_credentials_follow_the_exchange_not_the_environment(self):
        """Kalshi does not share keys between demo and production."""
        k = kalshi(
            environment=TradingEnvironment.PAPER,
            read_production_market_data=True,
            event_credentials=KalshiCredentials(api_key_id="demo-key", private_key_path="d.key"),
            prod_credentials=KalshiCredentials(api_key_id="prod-key", private_key_path="p.key"),
        )
        assert k.active_credentials.api_key_id == "prod-key"

    def test_demo_data_uses_the_demo_identity(self):
        k = kalshi(
            environment=TradingEnvironment.PAPER,
            event_credentials=KalshiCredentials(api_key_id="demo-key", private_key_path="d.key"),
            prod_credentials=KalshiCredentials(api_key_id="prod-key", private_key_path="p.key"),
        )
        assert k.active_credentials.api_key_id == "demo-key"


class TestTheFlagCannotArmTrading:
    @pytest.mark.parametrize("flag", [True, False])
    def test_paper_never_submits_real_orders_either_way(self, flag):
        s = AppSettings(
            environment=TradingEnvironment.PAPER,
            live_trading_armed=False,
            kalshi=kalshi(
                environment=TradingEnvironment.PAPER, read_production_market_data=flag
            ),
        )
        assert s.can_submit_real_orders is False

    def test_the_flag_alone_does_not_change_the_trading_environment(self):
        s = AppSettings(
            environment=TradingEnvironment.PAPER,
            kalshi=kalshi(
                environment=TradingEnvironment.PAPER, read_production_market_data=True
            ),
        )
        assert s.environment is TradingEnvironment.PAPER
        assert s.can_submit_real_orders is False

    def test_production_credentials_are_redacted_in_the_config_dump(self):
        s = AppSettings(
            kalshi=kalshi(
                prod_credentials=KalshiCredentials(
                    api_key_id="aaaabbbb-cccc-dddd-eeee-ffff00001111",
                    private_key_path="C:/secrets/prod.key",
                )
            )
        )
        dumped = s.redacted()["kalshi"]["prod_credentials"]
        assert dumped["api_key_id"] != "aaaabbbb-cccc-dddd-eeee-ffff00001111"
        assert dumped["private_key_path"] == "<configured>"
