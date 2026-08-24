"""REST client for Kalshi event contracts.

Thin, typed wrappers over the endpoints this platform actually uses. Endpoints
are named after their OpenAPI ``operationId`` so they can be checked against the
spec in ``docs/vendor/kalshi/openapi.yaml`` without guesswork.

Order creation deliberately does *not* live behind a retry. See
:meth:`EventContractsREST.create_order` and :class:`kat.kalshi.http.AmbiguousSubmission`.
"""

from __future__ import annotations

from typing import Any

from kat.common.enums import OrderAction, OutcomeSide, SelfTradePrevention, TimeInForce
from kat.common.fixedpoint import format_count, format_dollars
from kat.kalshi.event_contracts.models import (
    EventBalance,
    EventFill,
    EventMarket,
    EventOrder,
    EventPositionSnapshot,
    parse_orderbook,
)
from kat.kalshi.http import KalshiHTTPClient
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["EventContractsREST"]


class EventContractsREST:
    """Event-contract endpoints. One instance per process."""

    def __init__(self, http: KalshiHTTPClient) -> None:
        self.http = http

    # -- exchange / account --------------------------------------------------

    async def get_exchange_status(self) -> dict[str, Any]:
        """``GetExchangeStatus``. Unauthenticated; used as a liveness probe."""
        return await self.http.get("/exchange/status", authenticated=False)

    async def get_account_limits(self) -> dict[str, Any]:
        """``GetAccountApiLimits`` -- real tier budgets for the rate limiter."""
        return await self.http.get("/account/limits")

    async def get_endpoint_costs(self) -> dict[str, int]:
        """``ListNonDefaultEndpointCosts`` -- authoritative non-default costs."""
        payload = await self.http.get("/account/endpoint_costs")
        costs: dict[str, int] = {}
        for entry in payload.get("endpoint_costs") or payload.get("costs") or []:
            method = (entry.get("method") or "GET").upper()
            path = entry.get("path") or entry.get("endpoint")
            cost = entry.get("cost") or entry.get("token_cost")
            if path and cost:
                costs[f"{method} {path}"] = int(cost)
        return costs

    # -- market data ---------------------------------------------------------

    async def get_markets(
        self,
        *,
        status: str | None = "open",
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        limit: int = 200,
        max_pages: int = 50,
    ) -> list[EventMarket]:
        """``GetMarkets``. Paginated; returns parsed markets."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        raw = await self.http.paginate(
            "/markets", key="markets", params=params, limit=limit, max_pages=max_pages
        )
        return [EventMarket.parse(m) for m in raw]

    async def get_market(self, ticker: str) -> EventMarket:
        payload = await self.http.get(f"/markets/{ticker}")
        return EventMarket.parse(payload["market"])

    async def get_orderbook(self, ticker: str, *, depth: int | None = None) -> OrderBook:
        """``GetMarketOrderbook``. Returns a YES-denominated two-sided book."""
        params = {"depth": depth} if depth else None
        payload = await self.http.get(f"/markets/{ticker}/orderbook", params=params)
        return parse_orderbook(ticker, payload)

    async def get_orderbooks(self, tickers: list[str]) -> dict[str, OrderBook]:
        """``GetMarketOrderbooks`` -- several books in one request.

        Used at start-up to seed state for a watchlist without spending one
        rate-limit charge per market.
        """
        payload = await self.http.get(
            "/markets/orderbooks", params={"tickers": ",".join(tickers)}
        )
        out: dict[str, OrderBook] = {}
        for entry in payload.get("orderbooks") or []:
            ticker = entry.get("ticker")
            if ticker:
                out[ticker] = parse_orderbook(ticker, entry)
        return out

    async def get_trades(self, *, ticker: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        payload = await self.http.get("/markets/trades", params=params)
        return payload.get("trades") or []

    async def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        *,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> list[dict[str, Any]]:
        """``GetMarketCandlesticks``. ``period_interval`` is in minutes."""
        payload = await self.http.get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        return payload.get("candlesticks") or []

    async def get_events(self, *, status: str | None = "open", limit: int = 200) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        return await self.http.paginate("/events", key="events", params=params, limit=limit)

    async def get_series(self, series_ticker: str) -> dict[str, Any]:
        payload = await self.http.get(f"/series/{series_ticker}")
        return payload.get("series") or payload

    # -- portfolio -----------------------------------------------------------

    async def get_balance(self) -> EventBalance:
        return EventBalance.parse(await self.http.get("/portfolio/balance"))

    async def get_positions(self, *, limit: int = 200) -> list[EventPositionSnapshot]:
        raw = await self.http.paginate(
            "/portfolio/positions", key="market_positions", limit=limit
        )
        return [EventPositionSnapshot.parse(p) for p in raw]

    async def get_orders(
        self,
        *,
        status: str | None = None,
        ticker: str | None = None,
        limit: int = 200,
    ) -> list[EventOrder]:
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if ticker:
            params["ticker"] = ticker
        raw = await self.http.paginate("/portfolio/orders", key="orders", params=params, limit=limit)
        return [EventOrder.parse(o) for o in raw]

    async def get_order(self, order_id: str) -> EventOrder:
        payload = await self.http.get(f"/portfolio/orders/{order_id}")
        return EventOrder.parse(payload["order"])

    async def get_fills(
        self,
        *,
        ticker: str | None = None,
        order_id: str | None = None,
        limit: int = 200,
    ) -> list[EventFill]:
        params: dict[str, Any] = {}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        raw = await self.http.paginate("/portfolio/fills", key="fills", params=params, limit=limit)
        return [EventFill.parse(f) for f in raw]

    async def get_settlements(self, *, limit: int = 200) -> list[dict[str, Any]]:
        return await self.http.paginate("/portfolio/settlements", key="settlements", limit=limit)

    # -- order entry ---------------------------------------------------------

    async def create_order(
        self,
        *,
        ticker: str,
        client_order_id: str,
        outcome_side: OutcomeSide,
        action: OrderAction,
        count_qc: int,
        price_cc: int | None = None,
        time_in_force: TimeInForce = TimeInForce.GOOD_TILL_CANCELED,
        post_only: bool = False,
        reduce_only: bool = False,
        self_trade_prevention: SelfTradePrevention = SelfTradePrevention.TAKER_AT_CROSS,
        expiration_ts: int | None = None,
    ) -> EventOrder:
        """``CreateOrder``.

        **Not retried.** A timeout here raises
        :class:`kat.kalshi.http.AmbiguousSubmission` carrying ``client_order_id``;
        the execution engine resolves the true outcome by querying the exchange
        for that id. Retrying instead would risk a doubled position.

        ``count_qc`` is centi-contracts and is sent as ``count_fp``. Prices are
        sent as fixed-point dollar strings on the side named by ``outcome_side``,
        because that is the price the order is actually quoted at.
        """
        body: dict[str, Any] = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": outcome_side.value,
            "action": action.value,
            "count_fp": format_count(count_qc),
            "time_in_force": time_in_force.value,
            "self_trade_prevention_type": self_trade_prevention.value,
        }
        if price_cc is not None:
            key = "yes_price_dollars" if outcome_side is OutcomeSide.YES else "no_price_dollars"
            body[key] = format_dollars(price_cc)
        if post_only:
            body["post_only"] = True
        if reduce_only:
            body["reduce_only"] = True
        if expiration_ts is not None:
            body["expiration_ts"] = expiration_ts

        payload = await self.http.request(
            "POST",
            "/portfolio/orders",
            json_body=body,
            is_write=True,
            idempotent=False,
            client_order_id=client_order_id,
        )
        return EventOrder.parse(payload["order"])

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """``CancelOrder``. Idempotent: cancelling twice is harmless."""
        return await self.http.request(
            "DELETE", f"/portfolio/orders/{order_id}", is_write=True, idempotent=True
        )

    async def amend_order(
        self,
        order_id: str,
        *,
        client_order_id: str,
        count_qc: int,
        price_cc: int,
        outcome_side: OutcomeSide,
    ) -> dict[str, Any]:
        """``AmendOrderV2``. Treated as non-idempotent, like a create."""
        key = "yes_price_dollars" if outcome_side is OutcomeSide.YES else "no_price_dollars"
        return await self.http.request(
            "POST",
            f"/portfolio/events/orders/{order_id}/amend",
            json_body={
                "client_order_id": client_order_id,
                "count_fp": format_count(count_qc),
                key: format_dollars(price_cc),
            },
            is_write=True,
            idempotent=False,
            client_order_id=client_order_id,
        )

    async def decrease_order(self, order_id: str, *, reduce_by_qc: int) -> dict[str, Any]:
        """``DecreaseOrderV2``. Only ever shrinks an order, so it is safe to repeat."""
        return await self.http.request(
            "POST",
            f"/portfolio/events/orders/{order_id}/decrease",
            json_body={"reduce_by_fp": format_count(reduce_by_qc)},
            is_write=True,
            idempotent=True,
        )
