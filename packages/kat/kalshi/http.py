"""Shared HTTP transport for both Kalshi domains.

The important behaviour in this module is what it refuses to do.

Retries and idempotency
-----------------------
A GET is safe to repeat. An order submission is not: if the request left this
process and the response never arrived, the order may well be resting on the
book, and sending it again risks a double position. This client therefore
classifies every request as idempotent or not, and **never retries a
non-idempotent request whose outcome is unknown**. Instead it raises
:class:`AmbiguousSubmission`, which the execution engine resolves by querying the
exchange for the ``client_order_id`` it already assigned -- turning an unknown
outcome into a known one rather than guessing.

The one retry that is always safe is a 429: Kalshi documents that a rate-limited
request is rejected outright, so it never reached the matching engine.

Connection reuse
----------------
A single ``httpx.AsyncClient`` is held open for the process lifetime with
keep-alive and HTTP/2, so the order path pays no TCP or TLS handshake.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Literal

import httpx
import orjson

from kat.common.errors import ExchangeError, RateLimited
from kat.common.ids import current_correlation_id
from kat.kalshi.auth import KalshiSigner
from kat.kalshi.forbidden import assert_endpoint_permitted
from kat.kalshi.ratelimit import RateLimiter
from kat.observability.logging import get_logger
from kat.observability.metrics import (
    M_REST_ERRORS,
    M_REST_LATENCY,
    M_REST_REQUESTS,
    metrics,
)

log = get_logger(__name__)

__all__ = ["AmbiguousSubmission", "KalshiHTTPClient"]

Method = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

# Statuses worth retrying for an *idempotent* request. 502/503/504 are transient
# infrastructure; 500 is not retried because a deterministic server error will
# simply fail again and the retry only adds latency.
_RETRYABLE_STATUSES = frozenset({502, 503, 504})


class AmbiguousSubmission(ExchangeError):
    """A non-idempotent request may or may not have been applied.

    Carries the ``client_order_id`` so the caller can resolve the outcome by
    querying the exchange instead of retrying.
    """

    def __init__(self, message: str, *, client_order_id: str | None = None) -> None:
        super().__init__(message, retryable=False)
        self.client_order_id = client_order_id


class KalshiHTTPClient:
    """Authenticated, rate-limited REST client for one Kalshi domain.

    One instance per domain (event contracts, perpetuals) because each has its
    own credentials, its own base path and its own rate-limit buckets.
    """

    def __init__(
        self,
        *,
        base_url: str,
        signer: KalshiSigner | None,
        rate_limiter: RateLimiter,
        is_perp: bool,
        timeout_s: float = 8.0,
        connect_timeout_s: float = 4.0,
        max_retries: int = 3,
        user_agent: str = "kalshi-autonomous-trader/0.1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.rate_limiter = rate_limiter
        self.is_perp = is_perp
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            follow_redirects=False,
        )
        self._closed = False

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client.aclose()

    # -- core request --------------------------------------------------------

    async def request(
        self,
        method: Method,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        is_write: bool = False,
        idempotent: bool | None = None,
        client_order_id: str | None = None,
        authenticated: bool = True,
        max_wait_s: float | None = None,
    ) -> Any:
        """Issue one request, returning the decoded JSON body.

        Parameters
        ----------
        idempotent:
            Whether repeating this request is safe. Defaults to ``True`` for
            ``GET``/``DELETE`` and ``False`` for everything else. Order creation
            must leave this at its default.
        client_order_id:
            Attached to :class:`AmbiguousSubmission` so the caller can reconcile.
        max_wait_s:
            Give up rather than wait longer than this for rate-limit tokens.
        """
        # Before authentication, before the rate limiter, before anything that
        # could partially happen. A refused call never even produces a signature.
        assert_endpoint_permitted(method, path)

        if idempotent is None:
            idempotent = method in ("GET", "DELETE")

        bucket = self.rate_limiter.bucket_for(is_perp=self.is_perp, is_write=is_write)
        cost = self.rate_limiter.cost_for(method, path)

        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            attempt += 1

            if not await self.rate_limiter.acquire(bucket, cost, max_wait_s=max_wait_s):
                raise RateLimited(
                    f"local rate limiter declined {method} {path} within {max_wait_s}s",
                    bucket=bucket,
                )

            headers = self._auth_headers(method, path) if authenticated else {}
            correlation = current_correlation_id()
            if correlation:
                # Echoed back in our own logs; Kalshi ignores unknown headers.
                headers["X-Correlation-Id"] = correlation

            content = orjson.dumps(json_body) if json_body is not None else None
            if content is not None:
                headers["Content-Type"] = "application/json"

            try:
                with metrics.timer(M_REST_LATENCY, method=method, domain=self._domain):
                    response = await self._client.request(
                        method, path, params=params, content=content, headers=headers
                    )
                metrics.increment(M_REST_REQUESTS, method=method, domain=self._domain)

            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # The request never reached Kalshi, so it is safe to retry even
                # when it is not idempotent: nothing could have been applied.
                last_error = exc
                metrics.increment(M_REST_ERRORS, kind="connect", domain=self._domain)
                if attempt > self.max_retries:
                    break
                await self._backoff(attempt)
                continue

            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError) as exc:
                # The request may have been received and applied. This is the
                # ambiguous case that must never be blindly retried.
                metrics.increment(M_REST_ERRORS, kind="ambiguous", domain=self._domain)
                if not idempotent:
                    log.error(
                        "kalshi.ambiguous_submission",
                        method=method, path=path, error=str(exc),
                        client_order_id=client_order_id,
                    )
                    raise AmbiguousSubmission(
                        f"{method} {path} timed out after the request was sent; "
                        "outcome unknown and the request must not be repeated",
                        client_order_id=client_order_id,
                    ) from exc
                last_error = exc
                if attempt > self.max_retries:
                    break
                await self._backoff(attempt)
                continue

            except httpx.HTTPError as exc:
                last_error = exc
                metrics.increment(M_REST_ERRORS, kind="transport", domain=self._domain)
                if not idempotent:
                    raise AmbiguousSubmission(
                        f"{method} {path} failed in transport: {exc}",
                        client_order_id=client_order_id,
                    ) from exc
                if attempt > self.max_retries:
                    break
                await self._backoff(attempt)
                continue

            # --- got an HTTP response ---
            if response.status_code == 429:
                # Documented as carrying no penalty: the request was rejected
                # before execution, so retrying is safe for any method.
                self.rate_limiter.note_429(bucket)
                last_error = RateLimited(f"{method} {path}", bucket=bucket)
                if attempt > self.max_retries:
                    break
                await self._backoff(attempt, base=0.05)
                continue

            if response.status_code in _RETRYABLE_STATUSES:
                metrics.increment(
                    M_REST_ERRORS, kind=str(response.status_code), domain=self._domain
                )
                if not idempotent:
                    raise AmbiguousSubmission(
                        f"{method} {path} returned {response.status_code}; "
                        "outcome unknown and the request must not be repeated",
                        client_order_id=client_order_id,
                    )
                last_error = ExchangeError(
                    f"{method} {path} -> {response.status_code}",
                    status=response.status_code,
                    retryable=True,
                )
                if attempt > self.max_retries:
                    break
                await self._backoff(attempt)
                continue

            if response.status_code >= 400:
                raise self._error_from(response, method, path)

            return self._decode(response, method, path)

        # Named by type as well as message. httpx's timeout exceptions carry an
        # empty str(), so interpolating only the message produced "GET /events
        # failed after 4 attempts: " -- an error that reports a failure and
        # withholds every fact about it. A read timeout and a 502 are different
        # problems with different fixes, and for a while these were the same
        # line in the log.
        detail = f"{type(last_error).__name__}: {last_error}" if last_error else "no error recorded"
        raise ExchangeError(
            f"{method} {path} failed after {attempt} attempts: {detail}",
            retryable=True,
        ) from last_error

    # -- helpers -------------------------------------------------------------

    @property
    def _domain(self) -> str:
        return "perp" if self.is_perp else "event"

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if self.signer is None:
            raise ExchangeError(
                f"{self._domain} credentials are not configured; "
                "cannot make an authenticated request"
            )
        # Kalshi signs the full path from the API root, so the base URL's own
        # path prefix (/trade-api/v2) must be included.
        full_path = httpx.URL(self.base_url).path.rstrip("/") + path
        return self.signer.headers(method, full_path)

    @staticmethod
    async def _backoff(attempt: int, *, base: float = 0.2) -> None:
        """Exponential backoff with full jitter, capped.

        Jitter matters here: without it, several reconnecting workers retry in
        lockstep and recreate the burst that caused the failure.
        """
        delay = min(base * (2 ** (attempt - 1)), 5.0)
        await asyncio.sleep(random.uniform(0, delay))

    def _error_from(self, response: httpx.Response, method: str, path: str) -> ExchangeError:
        body = response.text[:2000]
        code = None
        message = body
        try:
            parsed = orjson.loads(response.content)
            if isinstance(parsed, dict):
                code = parsed.get("code") or parsed.get("error")
                message = parsed.get("message") or parsed.get("error") or body
        except Exception:
            pass
        metrics.increment(M_REST_ERRORS, kind=str(response.status_code), domain=self._domain)
        log.warning(
            "kalshi.http_error",
            method=method, path=path, status=response.status_code, code=code, message=message[:400],
        )
        return ExchangeError(
            f"{method} {path} -> {response.status_code}: {message}",
            status=response.status_code,
            code=code,
            retryable=False,
            body=body,
        )

    @staticmethod
    def _decode(response: httpx.Response, method: str, path: str) -> Any:
        if not response.content:
            return {}
        try:
            return orjson.loads(response.content)
        except orjson.JSONDecodeError as exc:
            # A malformed body from a 2xx means the schema changed or a proxy
            # interfered. Either way it is not something to paper over.
            raise ExchangeError(
                f"{method} {path} returned status {response.status_code} with an "
                f"undecodable body: {exc}",
                status=response.status_code,
                body=response.text[:500],
            ) from exc

    # -- convenience ---------------------------------------------------------

    async def get(self, path: str, **kw: Any) -> Any:
        return await self.request("GET", path, **kw)

    async def post(self, path: str, **kw: Any) -> Any:
        return await self.request("POST", path, is_write=True, **kw)

    async def delete(self, path: str, **kw: Any) -> Any:
        return await self.request("DELETE", path, is_write=True, **kw)

    async def paginate(
        self,
        path: str,
        *,
        key: str,
        params: dict[str, Any] | None = None,
        limit: int = 200,
        max_pages: int = 200,
    ) -> list[dict[str, Any]]:
        """Follow Kalshi's cursor pagination and return every item.

        ``max_pages`` is a guard, not a preference: a cursor bug that never
        advances would otherwise loop forever and drain the rate-limit bucket.
        """
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            page_params = dict(params or {})
            page_params["limit"] = limit
            if cursor:
                page_params["cursor"] = cursor
            payload = await self.get(path, params=page_params)
            items = payload.get(key) or []
            out.extend(items)
            cursor = payload.get("cursor") or None
            if not cursor or not items:
                break
        return out
