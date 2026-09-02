"""Bank the history of markets that have already settled.

Training a model to predict *settlement* needs markets whose outcome is known
and whose prices were recorded on the way there. The platform only records
prices for markets it is currently watching, so that dataset grew at the rate
the engine ran: 353 outcomes after a week, of which so few cleared the quote
minimum that walk-forward purging emptied the validation window entirely. The
model that best suits this venue was the one there was least data for.

None of that history had to be waited for. Kalshi keeps per-minute candlesticks
for every market, settled ones included, and each candle carries the bid, the
ask, the traded price and the volume. A settled market therefore offers exactly
what supervised learning wants and the platform could not previously obtain: a
full price history with a known label at the end of it. One settled baseball
game yields 710 quoted minutes and 3.3 million contracts of volume.

This writes that history into ``quotes`` and ``settlements``, the same tables the
live ingester fills, so ``train_model.py --label settlement`` consumes it without
knowing where it came from.

Two limits worth stating plainly. A candle is a one-minute summary rather than a
tape, so the recorded bid and ask are period closes and the intra-minute
sequence is lost -- fine for features sampled at a minute, wrong for anything
reasoning about queue position or microstructure. And resting *sizes* are not
published per candle, so ``bid_size_qc``/``ask_size_qc`` stay null here rather
than being invented; a strategy that needs depth must not be trained on this.

Not every settled market is worth banking. Kalshi auto-generates multivariate
parlay shards that settle without ever trading, and range series carry a strike
for every level including those nobody quoted. Markets whose candles show no
two-sided quote are skipped, because a price history with no prices in it is
label noise wearing the costume of a sample.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
import sys
from typing import Any

from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.config.settings import AppSettings
from kat.kalshi.auth import KalshiSigner
from kat.kalshi.event_contracts.rest import EventContractsREST
from kat.kalshi.http import KalshiHTTPClient
from kat.kalshi.ratelimit import RateLimiter
from kat.observability.logging import configure_logging

#: Series that settle often enough to build a dataset from. Long-dated markets
#: -- elections, awards -- have the volume but produce one label a year.
DEFAULT_SERIES = (
    "KXMLBGAME", "KXMLBF5", "KXMLBF3", "KXNFLGAME", "KXCFBGAME",
    "KXNBAGAME", "KXNHLGAME", "KXWNBAGAME", "KXEPLGAME", "KXUCLGAME",
    "KXBTCD", "KXETHD", "KXHIGHNY", "KXHIGHCHI", "KXHIGHLAX", "KXHIGHMIA",
)

#: How much of a market's life to bank, counting back from its close.
DEFAULT_HOURS = 24

#: A market needs this many minutes with a two-sided quote to be worth keeping.
#: Below it the "history" is a handful of prints around an untraded strike.
MIN_QUOTED_MINUTES = 30


def _cc(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value) * CC_PER_DOLLAR)
    except (TypeError, ValueError):
        return None


def _qc(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value) * 100)
    except (TypeError, ValueError):
        return None


def _ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


async def _client() -> KalshiHTTPClient:
    settings = AppSettings()
    signer = KalshiSigner.from_credentials(
        settings.kalshi.active_credentials, label="event"
    )
    return KalshiHTTPClient(
        base_url=settings.kalshi.rest_base_url,
        signer=signer,
        rate_limiter=RateLimiter(),
        is_perp=False,
        timeout_s=settings.kalshi.request_timeout_s,
        connect_timeout_s=settings.kalshi.connect_timeout_s,
    )


async def settled_markets(
    http: KalshiHTTPClient, series: str, *, max_pages: int
) -> list[dict[str, Any]]:
    """Every settled market in a series that actually resolved either way."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "series_ticker": series, "status": "settled", "limit": 200
        }
        if cursor:
            params["cursor"] = cursor
        try:
            payload = await http.get("/markets", params=params)
        except Exception as exc:
            print(f"  {series}: {type(exc).__name__}", file=sys.stderr)
            break
        markets = payload.get("markets") or []
        out.extend(m for m in markets if m.get("result") in ("yes", "no"))
        cursor = payload.get("cursor")
        if not cursor or not markets:
            break
    return out


async def backfill_market(
    conn: Any, rest: EventContractsREST, series: str, market: dict[str, Any],
    *, hours: int,
) -> int:
    """Write one settled market's candle history. Returns quoted minutes kept."""
    ticker = market.get("ticker")
    closed = _ts(market.get("close_time"))
    if not ticker or closed is None:
        return 0

    end_ts = int(closed.timestamp())
    try:
        candles = await rest.get_candlesticks(
            series, ticker,
            start_ts=end_ts - hours * 3600, end_ts=end_ts, period_interval=1,
        )
    except Exception:
        return 0

    rows = []
    for candle in candles:
        ts = candle.get("end_period_ts")
        bid = _cc((candle.get("yes_bid") or {}).get("close_dollars"))
        ask = _cc((candle.get("yes_ask") or {}).get("close_dollars"))
        # Both sides or nothing. A one-sided book cannot be priced, and a
        # feature built from half of one is a number with no market behind it.
        if not ts or bid is None or ask is None or not 0 < bid < ask < CC_PER_DOLLAR:
            continue
        period_end = dt.datetime.fromtimestamp(int(ts), tz=dt.UTC)
        rows.append((
            ticker, "event", period_end,
            bid, ask,
            _cc((candle.get("price") or {}).get("close_dollars")),
            _qc(candle.get("volume_fp")),
            _qc(candle.get("open_interest_fp")),
            period_end,
        ))

    if len(rows) < MIN_QUOTED_MINUTES:
        return 0

    await conn.executemany(
        """
        insert into quotes
          (ticker, market_type, ts, bid_cc, ask_cc, last_cc,
           volume_qc, open_interest_qc, source_ts, source, ingested_ts,
           quality)
        values ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'kalshi_candlestick', now(),
                'partial')
        """,
        rows,
    )
    # Quality is 'partial', not 'ok' and not 'synthetic'. These are real
    # exchange observations, so calling them synthetic would be false and would
    # bar them from a feed they belong in -- but they are a one-minute summary
    # with no resting sizes, so calling them complete would be equally false.
    # Tagged 'kalshi_candlestick' rather than left to look like live ingestion.
    # ingested_ts is honestly now, so the information-time rule in IngestMixin
    # keeps these rows out of any backtest positioned before today -- which is
    # correct, because we did not know them then. Training is a different
    # question and reads ts, with walk-forward splits doing the ordering.
    #
    # The label. settled_ts is the close, which is when the price history stops
    # and the only timestamp both the market and the candles agree on.
    await conn.execute(
        """
        insert into settlements
          (ticker, settled_ts, result, raw)
        values ($1, $2, $3, $4)
        on conflict on constraint uq_settlement do nothing
        """,
        ticker, closed, market["result"],
        # settlements carries no provenance columns of its own, so where this
        # came from goes in raw rather than nowhere.
        json.dumps({"source": "kalshi_candlestick", "series": series}),
    )
    return len(rows)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="*", default=list(DEFAULT_SERIES))
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    parser.add_argument("--max-pages", type=int, default=3,
                        help="pages of settled markets per series, 200 each")
    parser.add_argument("--limit-per-series", type=int, default=120)
    args = parser.parse_args()

    configure_logging("WARNING")
    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    http = await _client()
    rest = EventContractsREST(http)

    kept = skipped = quoted_minutes = 0
    try:
        have = {
            r["ticker"]
            for r in await conn.fetch("select distinct ticker from settlements")
        }
        for series in args.series:
            markets = await settled_markets(http, series, max_pages=args.max_pages)
            fresh = [m for m in markets if m.get("ticker") not in have]
            took = 0
            for market in fresh:
                if took >= args.limit_per_series:
                    break
                minutes = await backfill_market(
                    conn, rest, series, market, hours=args.hours
                )
                if minutes:
                    kept += 1
                    took += 1
                    quoted_minutes += minutes
                else:
                    skipped += 1
            print(f"  {series:<14} settled={len(markets):>4} new={len(fresh):>4} "
                  f"banked={took:>4}")
        print(f"\n  banked {kept} markets, {quoted_minutes:,} quoted minutes; "
              f"skipped {skipped} that never traded")
    finally:
        await conn.close()
        await http.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
