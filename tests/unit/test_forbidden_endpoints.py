"""The platform must be unable to touch money-movement endpoints.

Kalshi's own API makes withdrawal and deposit read-only, so an API key cannot
extract funds to a bank account in the first place. These tests cover the
remaining surface -- internal transfers and credential minting -- and make the
refusal architectural rather than a property of what today's code happens to
call.
"""

from __future__ import annotations

import pytest

from kat.kalshi.forbidden import (
    ForbiddenEndpoint,
    assert_endpoint_permitted,
    is_forbidden,
)
from kat.kalshi.http import KalshiHTTPClient
from kat.kalshi.ratelimit import RateLimiter


class TestDenyList:
    @pytest.mark.parametrize(
        "path",
        [
            "/portfolio/withdrawals",
            "/portfolio/deposits",
            "/portfolio/subaccounts",
            "/portfolio/subaccounts/transfer",
            "/portfolio/subaccounts/balances",
            "/portfolio/margin/subaccounts/transfer",
            "/portfolio/intra_exchange_instance_transfer",
            "/portfolio/intra_exchange_instance_transfers",
            "/api_keys",
            "/api_keys/generate",
            "/api_keys/abc-123",
        ],
    )
    def test_money_and_credential_endpoints_are_refused(self, path):
        assert is_forbidden(path) is not None
        with pytest.raises(ForbiddenEndpoint):
            assert_endpoint_permitted("POST", path)

    def test_reading_them_is_refused_too(self):
        """A GET on withdrawals is harmless, and still refused.

        The deny-list matches on path rather than method on purpose: an endpoint
        that gains a POST it does not have today should be refused by default,
        not admitted because the list was written when it was read-only.
        """
        with pytest.raises(ForbiddenEndpoint):
            assert_endpoint_permitted("GET", "/portfolio/withdrawals")

    @pytest.mark.parametrize(
        "path",
        [
            "/portfolio/balance",
            "/portfolio/orders",
            "/portfolio/positions",
            "/portfolio/fills",
            "/portfolio/settlements",
            "/markets",
            "/markets/KX-TEST/orderbook",
            "/exchange/status",
            "/margin/positions",
            "/margin/orders",
            "/margin/balance",
        ],
    )
    def test_the_endpoints_the_platform_actually_needs_are_permitted(self, path):
        assert is_forbidden(path) is None
        assert assert_endpoint_permitted("GET", path) is None

    def test_the_refusal_names_the_path_and_the_reason(self):
        with pytest.raises(ForbiddenEndpoint) as excinfo:
            assert_endpoint_permitted("POST", "/portfolio/subaccounts/transfer")
        message = str(excinfo.value)
        assert "/portfolio/subaccounts/transfer" in message
        assert "deny-list" in message
        assert excinfo.value.retryable is False


class TestEnforcedAtTheClient:
    async def test_the_http_client_refuses_before_signing(self):
        """No token acquired, no signature produced, no request attempted.

        The client is built with no signer at all: if the check ran after
        authentication this would raise some auth error instead, so reaching
        ForbiddenEndpoint proves the order.
        """
        client = KalshiHTTPClient(
            base_url="https://example.invalid/trade-api/v2",
            signer=None,
            rate_limiter=RateLimiter(),
            is_perp=False,
        )
        try:
            with pytest.raises(ForbiddenEndpoint):
                await client.request("POST", "/portfolio/subaccounts/transfer",
                                     json_body={"amount": 1}, is_write=True)
        finally:
            await client.aclose()

    async def test_a_permitted_path_gets_past_the_deny_list(self):
        """It fails later, for an unrelated reason -- which is the point."""
        client = KalshiHTTPClient(
            base_url="https://example.invalid/trade-api/v2",
            signer=None,
            rate_limiter=RateLimiter(),
            is_perp=False,
        )
        try:
            with pytest.raises(Exception) as excinfo:
                await client.request("GET", "/portfolio/balance")
            assert not isinstance(excinfo.value, ForbiddenEndpoint)
        finally:
            await client.aclose()
