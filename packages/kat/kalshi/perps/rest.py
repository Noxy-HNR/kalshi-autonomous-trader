"""REST client for Kalshi perpetuals (the ``/margin`` namespace).

Endpoints mirror the event-contract API but live under ``/margin`` and speak
fixed-point dollars throughout.

Not available on margin (documented, not an omission here): batch order
operations, queue positions, events/series/milestones, RFQs and quotes,
historical-data endpoints, and the exchange schedule. Where the platform needs
one of those for perps it uses the perp-specific alternative or does without;
:doc:`KALSHI.md <../../docs/KALSHI.md>` records each case.
"""

from __future__ import annotations

from typing import Any

from kat.common.enums import BookSide, SelfTradePrevention, TimeInForce
from kat.common.fixedpoint import format_count, format_dollars
from kat.kalshi.http import KalshiHTTPClient
from kat.kalshi.perps.models import (
    PerpBalance,
    PerpFill,
    PerpFundingRate,
    PerpMarket,
    PerpOrder,
    PerpPositionSnapshot,
    PerpRiskParameters,
    PerpRiskSnapshot,
    parse_margin_orderbook,
)
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["PerpsREST"]


class PerpsREST:
    """Perpetual-futures endpoints. One instance per process."""

    def __init__(self, http: KalshiHTTPClient) -> None:
        self.http = http

    # -- capability / status -------------------------------------------------

    async def is_margin_enabled(self) -> bool:
        """``GetMarginEnabled``.

        Perps roll out member by member, so this is checked at start-up and the
        entire perpetual subsystem stays dormant when it returns false rather
        than failing every request.
        """
        payload = await self.http.get("/margin/enabled")
        return bool(payload.get("enabled"))

    async def get_exchange_status(self) -> dict[str, Any]:
        return await self.http.get("/margin/exchange/status")

    async def get_account_limits(self) -> dict[str, Any]:
        """``GetPerpsAccountApiLimits`` -- perps have their own tier budgets."""
        return await self.http.get("/account/limits/perps")

    async def get_fee_tiers(self) -> dict[str, Any]:
        return await self.http.get("/margin/fee_tiers")

    # -- risk ----------------------------------------------------------------

    async def get_risk(self) -> PerpRiskSnapshot:
        """``GetMarginRisk``.

        Returns the exchange's leverage and liquidation view. Note that its
        ``account_leverage`` is notional / maintenance-margin and is *not* the
        1.00x invariant this platform enforces -- see :mod:`kat.risk.invariants`.
        """
        return PerpRiskSnapshot.parse(await self.http.get("/margin/risk"))

    async def get_risk_parameters(self) -> PerpRiskParameters:
        """``GetMarginRiskParameters`` -- liquidation thresholds, IM multipliers."""
        return PerpRiskParameters.parse(await self.http.get("/margin/risk_parameters"))

    async def get_notional_risk_limit(self) -> dict[str, Any]:
        """``GetMarginNotionalRiskLimit``.

        Read-only at the exchange: there is no endpoint to *set* it, so it acts
        as an additional ceiling we must respect rather than one we can tighten
        remotely. The local risk engine treats it as a hard cap alongside its own.
        """
        return await self.http.get("/margin/notional_risk_limit")

    # -- market data ---------------------------------------------------------

    async def get_markets(self, *, limit: int = 200, max_pages: int = 20) -> list[PerpMarket]:
        raw = await self.http.paginate(
            "/margin/markets", key="markets", limit=limit, max_pages=max_pages
        )
        return [PerpMarket.parse(m) for m in raw]

    async def get_market(self, ticker: str) -> PerpMarket:
        payload = await self.http.get(f"/margin/markets/{ticker}")
        return PerpMarket.parse(payload.get("market") or payload)

    async def get_orderbook(self, ticker: str, *, depth: int | None = None) -> OrderBook:
        params = {"depth": depth} if depth else None
        payload = await self.http.get(f"/margin/markets/{ticker}/orderbook", params=params)
        return parse_margin_orderbook(ticker, payload)

    async def get_candlesticks(
        self, ticker: str, *, start_ts: int, end_ts: int, period_interval: int = 1
    ) -> list[dict[str, Any]]:
        payload = await self.http.get(
            f"/margin/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        return payload.get("candlesticks") or []

    async def get_trades(self, *, ticker: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        payload = await self.http.get("/margin/trades", params=params)
        return payload.get("trades") or []

    # -- funding -------------------------------------------------------------

    async def get_funding_estimate(self, ticker: str) -> PerpFundingRate:
        payload = await self.http.get(
            "/margin/funding_rates/estimate", params={"ticker": ticker}
        )
        return PerpFundingRate.parse({"ticker": ticker, **payload}, is_estimate=True)

    async def get_historical_funding_rates(
        self, ticker: str, *, limit: int = 200
    ) -> list[PerpFundingRate]:
        payload = await self.http.get(
            "/margin/funding_rates/historical", params={"ticker": ticker, "limit": limit}
        )
        return [
            PerpFundingRate.parse({"ticker": ticker, **entry})
            for entry in payload.get("funding_rates") or []
        ]

    async def get_funding_history(self, *, limit: int = 200) -> list[dict[str, Any]]:
        payload = await self.http.get("/margin/funding_history", params={"limit": limit})
        return payload.get("funding_history") or payload.get("entries") or []

    # -- portfolio -----------------------------------------------------------

    async def get_balance(self, *, compute_available_balance: bool = False) -> PerpBalance:
        """``GetMarginBalance``.

        ``compute_available_balance`` costs 50 tokens instead of 5 because it
        scans every resting order, so it is requested on the reconciliation path
        and not on a fast poll.
        """
        params = {"compute_available_balance": "true"} if compute_available_balance else None
        return PerpBalance.parse(await self.http.get("/margin/balance", params=params))

    async def get_positions(self) -> list[PerpPositionSnapshot]:
        payload = await self.http.get("/margin/positions")
        return [PerpPositionSnapshot.parse(p) for p in payload.get("positions") or []]

    async def get_orders(self, *, ticker: str | None = None, limit: int = 200) -> list[PerpOrder]:
        params: dict[str, Any] = {}
        if ticker:
            params["ticker"] = ticker
        raw = await self.http.paginate("/margin/orders", key="orders", params=params, limit=limit)
        return [PerpOrder.parse(o) for o in raw]

    async def get_order(self, order_id: str) -> PerpOrder:
        payload = await self.http.get(f"/margin/orders/{order_id}")
        return PerpOrder.parse(payload.get("order") or payload)

    async def get_fills(self, *, ticker: str | None = None, limit: int = 200) -> list[PerpFill]:
        params: dict[str, Any] = {}
        if ticker:
            params["ticker"] = ticker
        raw = await self.http.paginate("/margin/fills", key="fills", params=params, limit=limit)
        return [PerpFill.parse(f) for f in raw]

    # -- order entry ---------------------------------------------------------

    async def create_order(
        self,
        *,
        ticker: str,
        client_order_id: str,
        side: BookSide,
        count_qc: int,
        price_cc: int,
        time_in_force: TimeInForce = TimeInForce.GOOD_TILL_CANCELED,
        post_only: bool = False,
        reduce_only: bool = False,
        self_trade_prevention: SelfTradePrevention = SelfTradePrevention.TAKER_AT_CROSS,
        subaccount: int = 0,
        cancel_order_on_pause: bool = True,
    ) -> PerpOrder:
        """``CreateMarginOrder``.

        **Not retried**, exactly as on the event side: an ambiguous outcome
        raises :class:`kat.kalshi.http.AmbiguousSubmission` for the execution
        engine to resolve by querying ``client_order_id``.

        ``cancel_order_on_pause`` defaults to true so that a resting perpetual
        order does not survive an exchange pause into a market whose price has
        moved -- the exposure it represents would otherwise be counted against a
        stale mark.

        Note that ``reduce_only`` is only accepted with an IOC or FOK
        time-in-force, so this method upgrades a reduce-only GTC to IOC rather
        than letting the exchange reject it.
        """
        if reduce_only and time_in_force is TimeInForce.GOOD_TILL_CANCELED:
            time_in_force = TimeInForce.IMMEDIATE_OR_CANCEL

        body: dict[str, Any] = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side.value,
            "count": format_count(count_qc),
            "price": format_dollars(price_cc),
            "time_in_force": time_in_force.value,
            "self_trade_prevention_type": self_trade_prevention.value,
            "subaccount": subaccount,
            "cancel_order_on_pause": cancel_order_on_pause,
        }
        if post_only:
            body["post_only"] = True
        if reduce_only:
            body["reduce_only"] = True

        payload = await self.http.request(
            "POST",
            "/margin/orders",
            json_body=body,
            is_write=True,
            idempotent=False,
            client_order_id=client_order_id,
        )
        return PerpOrder.parse(payload.get("order") or payload)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        return await self.http.request(
            "DELETE", f"/margin/orders/{order_id}", is_write=True, idempotent=True
        )

    async def decrease_order(self, order_id: str, *, reduce_by_qc: int) -> dict[str, Any]:
        """``DecreaseMarginOrder``. Only shrinks, so safe to repeat."""
        return await self.http.request(
            "POST",
            f"/margin/orders/{order_id}/decrease",
            json_body={"reduce_by": format_count(reduce_by_qc)},
            is_write=True,
            idempotent=True,
        )
