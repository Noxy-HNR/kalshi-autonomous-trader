"""Polymarket US REST access.

The exchange splits its API across two hosts, and the split is unusually useful:

``https://gateway.polymarket.us``
    Public. Markets, events, order books, best bid/offer, search. No key, no
    account, no KYC. Everything needed to *measure* the venue.

``https://api.polymarket.us``
    Authenticated. Orders, positions, balances, and the trade-statistics
    endpoint that serves candlesticks. Keys are issued only after identity
    verification through the exchange's iOS app.

So the research half of this platform can run against the real venue today and
the trading half cannot, which is the right way round: nothing here can place an
order, and the survey that decides whether the venue is worth trading needs no
permission to run.

What the public half cannot give
--------------------------------
Price history. ``/v1beta1/report/trades/stats`` is authenticated, and even with
a key its candles carry ``open/high/low/close/volume`` and no bid or ask. The
Kalshi work leaned heavily on candles that included both sides of the book,
which is what let a market-making strategy be backtested against history that
already existed. That shortcut does not exist here: measuring a resting quote on
this venue means banking books as they happen.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from kat.common.errors import ExchangeError
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger
from kat.polymarket.models import PolymarketMarket, parse_book, price_cc

__all__ = ["PUBLIC_GATEWAY", "PolymarketREST"]

log = get_logger(__name__)

PUBLIC_GATEWAY = "https://gateway.polymarket.us"

#: The exchange caps a market page; larger requests are silently truncated.
_MAX_PAGE = 500


class PolymarketREST:
    """Read-only access to the public gateway.

    Deliberately has no order methods. Placing an order needs the authenticated
    host and a verified account, and a client that *could* trade but has no
    credentials is a worse object than one that structurally cannot: the second
    cannot be handed to something that assumes it works.
    """

    def __init__(self, base_url: str = PUBLIC_GATEWAY, *, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # -- transport -----------------------------------------------------------

    def _get_sync(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(
            {k: v for k, v in (params or {}).items() if v is not None}
        )
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "kat/polymarket"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read()[:400].decode("utf-8", "replace")
            raise ExchangeError(f"GET {path} -> {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ExchangeError(f"GET {path} -> {exc.reason}") from exc

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Run the blocking fetch off the event loop.

        ``urllib`` rather than the platform's signed HTTP client because there is
        nothing to sign: the gateway takes no credentials, and routing an
        unauthenticated read through an authenticating client would invite one to
        be attached later.
        """
        return await asyncio.to_thread(self._get_sync, path, params)

    # -- markets -------------------------------------------------------------

    async def get_markets(
        self,
        *,
        closed: bool | None = None,
        active: bool | None = None,
        limit: int = _MAX_PAGE,
        offset: int = 0,
        categories: str | None = None,
    ) -> list[PolymarketMarket]:
        payload = await self._get(
            "/v1/markets",
            {
                "closed": None if closed is None else str(closed).lower(),
                "active": None if active is None else str(active).lower(),
                "limit": min(limit, _MAX_PAGE),
                "offset": offset,
                "categories": categories,
            },
        )
        return [
            PolymarketMarket.parse(m) for m in (payload.get("markets") or [])
        ]

    async def get_open_markets(
        self, *, max_pages: int = 10, categories: str | None = None
    ) -> list[PolymarketMarket]:
        """Every market currently open for trading.

        Pages by offset because the endpoint publishes no cursor. A short page
        ends the walk; the default listing returns *resolved* markets, so the
        open filter is not optional.
        """
        out: list[PolymarketMarket] = []
        for page in range(max_pages):
            batch = await self.get_markets(
                closed=False, active=True,
                limit=_MAX_PAGE, offset=page * _MAX_PAGE, categories=categories,
            )
            out.extend(batch)
            if len(batch) < _MAX_PAGE:
                break
        return out

    async def get_market(self, slug: str) -> PolymarketMarket:
        payload = await self._get(f"/v1/markets/{slug}")
        market = payload.get("market") or payload
        return PolymarketMarket.parse(market)

    # -- book ----------------------------------------------------------------

    async def get_book(self, slug: str, *, ts_ms: int = 0) -> OrderBook | None:
        """Full order book, or None when it is one-sided."""
        payload = await self._get(f"/v1/markets/{slug}/book")
        return parse_book(slug, payload, ts_ms=ts_ms)

    async def get_bbo(self, slug: str) -> dict[str, Any]:
        """Top of book plus traded volume and open interest.

        ``bidDepth`` and ``askDepth`` here are *level counts*, not sizes, so
        this cannot replace :meth:`get_book` for anything that cares about the
        queue ahead of a resting order -- which is most of why we are here.
        What it adds is ``sharesTraded`` and ``lastTradePx``.
        """
        payload = await self._get(f"/v1/markets/{slug}/bbo")
        return payload.get("marketData") or payload

    async def get_flow(self, slug: str) -> tuple[int | None, int | None]:
        """``(cumulative shares traded in qc, last traded price in cc)``.

        The venue publishes no public trades endpoint -- ``/trades`` and
        ``/markets/{slug}/trades`` are both 404 -- so flow has to be recovered
        by differencing this counter between passes. Without it a banked book
        cannot be replayed: the fill simulator fills a resting order when the
        market trades *through* it, and a book with no prints beside it never
        fills anything, which would read as a strategy that never trades rather
        than as data that cannot answer the question.
        """
        data = await self.get_bbo(slug)
        traded = data.get("sharesTraded")
        try:
            shares = None if traded in (None, "") else round(float(traded) * 100)
        except (TypeError, ValueError):
            shares = None
        return shares, price_cc(data.get("lastTradePx"))

    # -- events --------------------------------------------------------------

    async def get_events(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        payload = await self._get("/v1/events", {"limit": limit, "offset": offset})
        return payload.get("events") or []
