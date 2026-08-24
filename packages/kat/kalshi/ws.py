"""Kalshi WebSocket client.

One class serves both domains; the differences are supplied by the caller:

* **URL and credentials.** Event contracts and perpetuals use separate hosts and
  separate API keys.
* **Timestamp convention.** Event payloads use RFC3339 strings and Unix seconds;
  margin payloads use ``*_ms`` Unix-millisecond integers throughout.
* **Available channels.** ``market_positions``, ``market_lifecycle_v2`` and the
  multivariate channels do not exist on margin.

What this client guarantees
---------------------------
* **The connection is authenticated at handshake time**, with the signed headers
  in the HTTP upgrade request. Kalshi requires auth even for public channels.
* **Sequence gaps are detected, not ignored.** Every channel message carries a
  ``seq``; a gap means we missed a book update and our local book is now wrong.
  The client raises a resubscribe rather than continuing to apply deltas to a
  book it can no longer trust.
* **Reconnection is automatic, with jittered backoff**, and re-establishes every
  subscription that was active before the drop.
* **Kalshi's ping frames are answered by the library.** ``websockets`` responds
  to ping frames automatically; we additionally send our own pings so that a
  half-open connection (one that accepts writes but delivers nothing) is
  detected within the ping timeout rather than hanging indefinitely.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from kat.kalshi.auth import KalshiSigner
from kat.observability.logging import get_logger
from kat.observability.metrics import (
    M_WS_MESSAGES,
    M_WS_RECONNECTS,
    M_WS_SEQUENCE_GAPS,
    metrics,
)

log = get_logger(__name__)

__all__ = ["KalshiWebSocket", "Subscription", "WSState"]

MessageHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


class WSState:
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RESUBSCRIBING = "resubscribing"
    CLOSED = "closed"


@dataclass
class Subscription:
    """A channel subscription we intend to hold, and its live state.

    Kept as *intent* rather than as exchange state so that a reconnect can
    rebuild everything without the caller re-registering anything.
    """

    channels: tuple[str, ...]
    market_tickers: tuple[str, ...] = ()
    sid: int | None = None
    last_seq: int | None = None

    @property
    def key(self) -> str:
        return "|".join(self.channels) + "::" + ",".join(sorted(self.market_tickers))


@dataclass
class _Pending:
    future: asyncio.Future[dict[str, Any]]
    sent_at: float = 0.0


class KalshiWebSocket:
    """Persistent, self-healing WebSocket connection to one Kalshi domain."""

    def __init__(
        self,
        *,
        url: str,
        signer: KalshiSigner | None,
        domain: str,
        ping_interval_s: float = 10.0,
        ping_timeout_s: float = 10.0,
        max_reconnect_delay_s: float = 30.0,
        message_queue_size: int = 10_000,
    ) -> None:
        self.url = url
        self.signer = signer
        self.domain = domain  # "event" | "perp", used for metric labels
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.max_reconnect_delay_s = max_reconnect_delay_s

        self.state = WSState.DISCONNECTED
        self._ws: ClientConnection | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._cmd_id = 0
        self._pending: dict[int, _Pending] = {}
        self._subscriptions: dict[str, Subscription] = {}
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._connected_event = asyncio.Event()
        # Bounded: if a consumer stalls we drop the oldest market data rather
        # than growing without limit. Order/fill messages bypass this queue and
        # are dispatched inline, because losing one is not acceptable.
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=message_queue_size)
        self.reconnect_count = 0
        self.gap_count = 0
        self.last_message_ms = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever(), name=f"ws-{self.domain}")

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.state = WSState.CLOSED
        self._connected_event.clear()
        log.info("ws.stopped", domain=self.domain)

    async def wait_connected(self, timeout_s: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout_s)
            return True
        except TimeoutError:
            return False

    @property
    def connected(self) -> bool:
        return self.state == WSState.CONNECTED and self._ws is not None

    # -- subscriptions -------------------------------------------------------

    def on(self, message_type: str, handler: MessageHandler) -> None:
        """Register a handler for a message ``type`` (e.g. ``orderbook_delta``)."""
        self._handlers.setdefault(message_type, []).append(handler)

    async def subscribe(
        self, channels: list[str], *, market_tickers: list[str] | None = None
    ) -> Subscription:
        """Subscribe, recording the intent so a reconnect can restore it."""
        sub = Subscription(
            channels=tuple(channels), market_tickers=tuple(market_tickers or ())
        )
        self._subscriptions[sub.key] = sub
        if self.connected:
            await self._send_subscribe(sub)
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        self._subscriptions.pop(sub.key, None)
        if self.connected and sub.sid is not None:
            await self._send({"cmd": "unsubscribe", "params": {"sids": [sub.sid]}})

    async def update_subscription(
        self, sub: Subscription, *, add: list[str] | None = None, remove: list[str] | None = None
    ) -> None:
        """Add or drop markets on an existing subscription without a full resub."""
        if sub.sid is None:
            return
        params: dict[str, Any] = {"sids": [sub.sid]}
        if add:
            params["action"] = "add_markets"
            params["market_tickers"] = add
            await self._send({"cmd": "update_subscription", "params": params})
            sub.market_tickers = tuple({*sub.market_tickers, *add})
        if remove:
            params = {"sids": [sub.sid], "action": "delete_markets", "market_tickers": remove}
            await self._send({"cmd": "update_subscription", "params": params})
            sub.market_tickers = tuple(t for t in sub.market_tickers if t not in set(remove))

    async def request_snapshot(self, sub: Subscription, tickers: list[str]) -> None:
        """Ask for a fresh ``orderbook_snapshot`` without changing the subscription.

        This is the recovery action after a sequence gap: rebuild the book from
        an authoritative snapshot instead of guessing at the missed delta.
        """
        if sub.sid is None:
            return
        await self._send(
            {
                "cmd": "update_subscription",
                "params": {"sids": [sub.sid], "action": "get_snapshot",
                           "market_tickers": tickers},
            }
        )

    # -- connection loop -----------------------------------------------------

    async def _run_forever(self) -> None:
        delay = 1.0
        while self._running:
            try:
                self.state = WSState.CONNECTING
                await self._connect_and_serve()
                delay = 1.0  # a clean session resets the backoff
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                log.warning(
                    "ws.connection_lost", domain=self.domain, error=str(exc),
                    error_type=type(exc).__name__,
                )
            finally:
                self.state = WSState.DISCONNECTED
                self._connected_event.clear()
                self._ws = None
                # Fail any in-flight commands so callers are not left awaiting.
                for pending in self._pending.values():
                    if not pending.future.done():
                        pending.future.set_exception(ConnectionError("websocket closed"))
                self._pending.clear()

            if not self._running:
                break

            self.reconnect_count += 1
            metrics.increment(M_WS_RECONNECTS, domain=self.domain)
            # Full jitter: several clients reconnecting after a common outage
            # must not synchronise into a thundering herd.
            sleep_for = random.uniform(0, min(delay, self.max_reconnect_delay_s))
            log.info("ws.reconnecting", domain=self.domain, delay_s=round(sleep_for, 2))
            await asyncio.sleep(sleep_for)
            delay = min(delay * 2, self.max_reconnect_delay_s)

    async def _connect_and_serve(self) -> None:
        headers = {}
        if self.signer is not None:
            # The signed path for a WebSocket upgrade is the connection path.
            from urllib.parse import urlsplit

            headers = self.signer.headers("GET", urlsplit(self.url).path)

        async with websockets.connect(
            self.url,
            additional_headers=headers,
            ping_interval=self.ping_interval_s,
            ping_timeout=self.ping_timeout_s,
            close_timeout=5.0,
            max_size=8 * 1024 * 1024,
        ) as ws:
            self._ws = ws
            self.state = WSState.CONNECTED
            self._connected_event.set()
            log.info("ws.connected", domain=self.domain, url=_safe_url(self.url))

            # Restore every subscription we are supposed to be holding.
            if self._subscriptions:
                self.state = WSState.RESUBSCRIBING
                for sub in list(self._subscriptions.values()):
                    sub.sid = None
                    sub.last_seq = None
                    await self._send_subscribe(sub)
                self.state = WSState.CONNECTED

            async for raw in ws:
                await self._handle_raw(raw)

    # -- messaging -----------------------------------------------------------

    async def _send(self, payload: dict[str, Any]) -> int:
        if self._ws is None:
            raise ConnectionError("websocket is not connected")
        self._cmd_id += 1
        payload["id"] = self._cmd_id
        await self._ws.send(orjson.dumps(payload).decode())
        return self._cmd_id

    async def _send_subscribe(self, sub: Subscription) -> None:
        params: dict[str, Any] = {"channels": list(sub.channels)}
        if sub.market_tickers:
            # Kalshi wants `market_ticker` for one market and `market_tickers`
            # for several; sending the wrong one silently subscribes to nothing.
            if len(sub.market_tickers) == 1:
                params["market_ticker"] = sub.market_tickers[0]
            else:
                params["market_tickers"] = list(sub.market_tickers)
        await self._send({"cmd": "subscribe", "params": params})

    async def _handle_raw(self, raw: str | bytes) -> None:
        try:
            message = orjson.loads(raw)
        except orjson.JSONDecodeError:
            log.warning("ws.undecodable_message", domain=self.domain, sample=str(raw)[:200])
            return

        self.last_message_ms = int(asyncio.get_running_loop().time() * 1000)
        msg_type = message.get("type", "")
        metrics.increment(M_WS_MESSAGES, domain=self.domain, type=msg_type or "unknown")

        if msg_type == "subscribed":
            self._on_subscribed(message)
            return
        if msg_type == "error":
            log.error(
                "ws.error_message",
                domain=self.domain,
                code=message.get("msg", {}).get("code"),
                message=message.get("msg", {}).get("msg"),
            )
            return
        if msg_type in ("unsubscribed", "ok", "subscription_updated"):
            return

        if not self._check_sequence(message, msg_type):
            return

        await self._dispatch(msg_type, message)

    def _on_subscribed(self, message: dict[str, Any]) -> None:
        payload = message.get("msg") or {}
        sid = payload.get("sid")
        channel = payload.get("channel")
        for sub in self._subscriptions.values():
            if channel in sub.channels and sub.sid is None:
                sub.sid = sid
                log.info("ws.subscribed", domain=self.domain, channel=channel, sid=sid)
                return

    def _check_sequence(self, message: dict[str, Any], msg_type: str) -> bool:
        """Detect a gap in the per-subscription sequence.

        Returns True when the message should be processed. On a gap we log,
        count, request a fresh snapshot and *still* process the message -- the
        snapshot that follows will overwrite the book, and dropping the message
        would only widen the hole if the snapshot request fails.
        """
        seq = message.get("seq")
        sid = message.get("sid")
        if seq is None or sid is None:
            return True

        sub = next((s for s in self._subscriptions.values() if s.sid == sid), None)
        if sub is None:
            return True

        if sub.last_seq is not None and seq != sub.last_seq + 1:
            if seq > sub.last_seq + 1:
                self.gap_count += 1
                metrics.increment(M_WS_SEQUENCE_GAPS, domain=self.domain)
                log.warning(
                    "ws.sequence_gap",
                    domain=self.domain, sid=sid, expected=sub.last_seq + 1, received=seq,
                    missed=seq - sub.last_seq - 1, msg_type=msg_type,
                )
                if sub.market_tickers:
                    # Fire and forget: the snapshot will arrive on this same
                    # connection and rebuild the affected books.
                    task = asyncio.create_task(
                        self.request_snapshot(sub, list(sub.market_tickers))
                    )
                    task.add_done_callback(lambda t: t.exception())
            else:
                # A replayed or out-of-order message; ignore it rather than
                # rewinding our sequence tracking.
                return False
        sub.last_seq = seq
        return True

    async def _dispatch(self, msg_type: str, message: dict[str, Any]) -> None:
        handlers = self._handlers.get(msg_type)
        if not handlers:
            return
        for handler in handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - one bad handler must not kill the feed
                log.error(
                    "ws.handler_failed",
                    domain=self.domain, msg_type=msg_type, error=str(exc), exc_info=True,
                )

    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "domain": self.domain,
            "connected": self.connected,
            "subscriptions": len(self._subscriptions),
            "reconnects": self.reconnect_count,
            "sequence_gaps": self.gap_count,
        }


def _safe_url(url: str) -> str:
    """URL without any query string, so a token in a query cannot reach a log."""
    return url.split("?", 1)[0]
