"""Build a backtest feed from Kalshi's own candlesticks.

The banked-quotes feed can only replay markets this platform has already
watched, and the watchlist was for a long time chosen by rules written for a
strategy that crosses the spread. So the most active markets on the exchange --
879 of them trade over a thousand contracts a day, 159 over ten thousand -- have
no local history at all, and the one structure that is positive before any
forecast, quoting passively where the spread covers both fees, could not be
measured on the markets it is actually for.

``GET /series/{s}/markets/{t}/candlesticks`` closes that gap. Each candle
carries the bid and ask over the period *and* the volume traded in it, which is
exactly the two things a fill simulator needs: a book to rest against and the
flow that reaches a resting order.

What this is and is not
-----------------------
A candle is a summary, not a tape. Within a minute it says what the bid and ask
were and how much traded, but not in what order, so the simulator sees one book
update and one print per period rather than the real sequence. That is coarser
than the websocket feed and it is honest about which direction the coarseness
cuts: a single print of a minute's volume reaches further into the queue than
the same volume split across many smaller trades would, so fills here are
*easier* than reality. Treat a result from this as an upper bound, and confirm
anything promising against banked quotes once the markets have been watched.

Prices are the close of each period. Volume is placed at the mid, because a
candle does not say which side traded.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

from kat.common.enums import MarketType
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.config.settings import AppSettings, EventRiskSettings
from kat.kalshi.auth import KalshiSigner
from kat.kalshi.event_contracts.rest import EventContractsREST
from kat.kalshi.http import KalshiHTTPClient
from kat.kalshi.ratelimit import RateLimiter
from kat.observability.logging import configure_logging
from kat.simulation.engine import (
    BacktestConfig,
    BacktestEngine,
    EventFeed,
    book_event,
    trade_event,
)
from kat.strategies.event.passive_quote import PassiveQuoteStrategy


class CandleQuoteStrategy(PassiveQuoteStrategy):
    """The quoting strategy, told the truth about how coarse this feed is.

    Live, the book arrives on a websocket and five seconds of silence means
    something is wrong, so the base class refuses to act on anything older. A
    candle is a one-minute summary: the book here is *always* up to a minute
    old, and that is the data's resolution rather than a fault. Leaving the live
    bar in place would make every decision a stale-data skip and measure
    nothing. This override is why results from this feed are an upper bound.
    """

    MAX_DATA_AGE_MS = 70_000


def _cc(value: Any) -> int | None:
    """Dollars-as-string to centicents, or None when absent."""
    if value in (None, ""):
        return None
    try:
        return round(float(value) * CC_PER_DOLLAR)
    except (TypeError, ValueError):
        return None


def _qc(value: Any) -> int:
    try:
        return round(float(value) * 100)
    except (TypeError, ValueError):
        return 0


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


async def pick_markets(rest: EventContractsREST, *, count: int, min_volume_qc: int):
    """The most-traded quoted markets whose spread covers both fees."""
    from kat.engine.ingestion import _maker_edge_cc

    markets = await rest.get_markets_via_events(status="open", limit=200, max_pages=25)
    live = [
        m
        for m in markets
        if m.is_tradeable
        and m.has_live_quote
        and (m.volume_24h_qc or 0) >= min_volume_qc
        and (_maker_edge_cc(m) or 0) > 0
        and m.series_ticker
    ]
    live.sort(key=lambda m: m.volume_24h_qc or 0, reverse=True)
    return live[:count]


async def build_feed(
    rest: EventContractsREST,
    markets: list[Any],
    *,
    hours: int,
    depth_qc: int = 10_000,
    low_share: float = 0.5,
    real_depth: dict[str, int] | None = None,
):
    """One book event and one print per candle, in time order."""
    end_ts = int(time.time())
    start_ts = end_ts - hours * 3600
    events: list[Any] = []
    seq = 0
    with_volume = 0

    for market in markets:
        # Each market's own queue, when we have it. A single synthetic depth for
        # every market is the assumption that decides the whole result: the
        # touch on these books ranges from one contract to over a million, and
        # a resting order's fate is entirely which of those it joined. The book
        # is a snapshot applied across the window, which is the honest limit of
        # this -- Kalshi publishes no historical depth -- but a real per-market
        # number beats an invented global one.
        market_depth = (real_depth or {}).get(market.ticker, depth_qc)
        try:
            candles = await rest.get_candlesticks(
                market.series_ticker, market.ticker,
                start_ts=start_ts, end_ts=end_ts, period_interval=1,
            )
        except Exception as exc:  # one market must not cost the run
            print(f"  {market.ticker}: {type(exc).__name__}", file=sys.stderr)
            continue

        for candle in candles:
            ts_ms = int(candle.get("end_period_ts", 0)) * 1000
            if ts_ms <= 0:
                continue
            bid = _cc((candle.get("yes_bid") or {}).get("close_dollars"))
            ask = _cc((candle.get("yes_ask") or {}).get("close_dollars"))
            if bid is None or ask is None or not 0 < bid < ask < CC_PER_DOLLAR:
                continue

            seq += 1
            events.append(
                book_event(
                    market.ticker,
                    info_ts_ms=ts_ms,
                    # Depth is not published per candle. One contract-lot at the
                    # touch is deliberately thin: it makes our own taker fills
                    # small and leaves the queue ahead of a resting order large.
                    bids=[(bid, market_depth)],
                    asks=[(ask, market_depth)],
                    seq=seq,
                )
            )

            volume_qc = _qc(candle.get("volume_fp"))
            if volume_qc > 0:
                with_volume += 1
                # At the prices that actually traded, not at the mid. Which
                # price a print lands on is the whole question for a resting
                # order: a bid is reached when somebody sells down to it, and a
                # print at the mid reaches nobody on either side. The candle
                # gives the traded low and high, so the volume is split between
                # them -- the low fills resting bids, the high fills resting
                # offers. Splitting evenly is a guess about which side was the
                # aggressor, which the candle does not record; it is the
                # conventional one and it does not favour us, because half the
                # flow is assigned to the side we are not quoting.
                price = candle.get("price") or {}
                low = _cc(price.get("low_dollars"))
                high = _cc(price.get("high_dollars"))
                prints: list[tuple[int, int]] = []
                if low is not None and high is not None and low != high:
                    at_low = int(volume_qc * low_share)
                    prints = [(low, at_low), (high, volume_qc - at_low)]
                elif low is not None:
                    prints = [(low, volume_qc)]
                elif high is not None:
                    prints = [(high, volume_qc)]
                for offset, (price_cc, qty_qc) in enumerate(prints):
                    if qty_qc <= 0 or not 0 < price_cc < CC_PER_DOLLAR:
                        continue
                    seq += 1
                    events.append(
                        trade_event(
                            market.ticker,
                            info_ts_ms=ts_ms + 1 + offset,
                            price_cc=price_cc,
                            qty_qc=qty_qc,
                            seq=seq,
                        )
                    )

    events.sort(key=lambda e: (e.info_ts_ms, e.seq))
    print(f"  feed     {len(events)} events, {with_volume} with volume, "
          f"{len(markets)} markets")
    return EventFeed(events, name="candles")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--markets", type=int, default=40)
    parser.add_argument("--min-volume-qc", type=int, default=1_000_000)
    parser.add_argument("--capital", type=float, default=200.0)
    parser.add_argument("--min-edge-cc", type=int, default=None)
    parser.add_argument("--min-price-cc", type=int, default=None)
    parser.add_argument("--max-price-cc", type=int, default=None)
    parser.add_argument("--max-spread-cc", type=int, default=None)
    parser.add_argument("--fee-cover-bps", type=int, default=None)
    parser.add_argument("--min-depth-qc", type=int, default=None)
    parser.add_argument("--depth-qc", type=int, default=10_000,
                        help="synthetic size at each touch; this is the queue "
                             "a resting order sits behind")
    parser.add_argument("--real-depth", action="store_true",
                        help="use each market's actual resting size at the "
                             "touch instead of one synthetic depth for all")
    parser.add_argument("--low-share", type=float, default=0.5,
                        help="fraction of a candle's volume assigned to the "
                             "traded low rather than the high")
    args = parser.parse_args()

    configure_logging("WARNING")
    http = await _client()
    rest = EventContractsREST(http)
    try:
        markets = await pick_markets(
            rest, count=args.markets, min_volume_qc=args.min_volume_qc
        )
        if not markets:
            print("no market meets the liquidity and spread bar", file=sys.stderr)
            return 1
        print(f"  markets  {len(markets)}, "
              f"top {markets[0].ticker} at {(markets[0].volume_24h_qc or 0)//100:,}/day")
        real_depth: dict[str, int] = {}
        if args.real_depth:
            for market in markets:
                try:
                    book = await rest.get_orderbook(market.ticker, depth=5)
                except Exception:
                    continue
                queue = min(book.best_bid_size_qc, book.best_ask_size_qc)
                if queue > 0:
                    real_depth[market.ticker] = queue
            sizes = sorted(real_depth.values())
            if sizes:
                print(f"  depth    real for {len(sizes)}/{len(markets)}, "
                      f"median {sizes[len(sizes) // 2] // 100:,} contracts at the touch")
        feed = await build_feed(
            rest, markets, hours=args.hours,
            depth_qc=args.depth_qc, low_share=args.low_share,
            real_depth=real_depth or None,
        )
    finally:
        await http.aclose()

    overrides = {
        k: v
        for k, v in (
            ("min_price_cc", args.min_price_cc),
            ("max_price_cc", args.max_price_cc),
            ("max_spread_cc", args.max_spread_cc),
            ("fee_cover_bps", args.fee_cover_bps),
            ("min_depth_qc", args.min_depth_qc),
        )
        if v is not None
    }
    if overrides:
        print(f"  params   {overrides}")

    risk = EventRiskSettings()
    if args.min_edge_cc is not None:
        risk = EventRiskSettings(min_edge_cc=args.min_edge_cc)
        print(f"  EXPERIMENT: min_edge_cc={args.min_edge_cc} for this run only")

    engine = BacktestEngine(
        feed=feed,
        strategies=[CandleQuoteStrategy(overrides)],
        config=BacktestConfig(
            market_type=MarketType.EVENT,
            initial_capital_micros=int(args.capital * 1_000_000),
            # Match the candle period: deciding more often than the data
            # changes just re-reads the same book.
            decision_interval_ms=60_000,
        ),
        settings=AppSettings(),
        event_risk_settings=risk,
    )
    result = engine.run()

    trades = getattr(result, "trades", None) or []
    distinct = {
        (t.ticker, t.entry_ms, t.exit_ms, t.qty_qc, t.net_pnl_micros) for t in trades
    }
    metrics = result.metrics
    print()
    print(f"  fills    {len(engine.fills.fills)}")
    print(f"  trades   {len(trades)} booked, {len(distinct)} distinct")
    print(f"  return   {metrics.total_return:+.4%}")
    print(f"  sharpe   {metrics.sharpe:.2f}")
    print(f"  win rate {metrics.win_rate:.0%}")
    print(f"  fees     ${metrics.total_fees_micros / 1e6:.4f}")
    print(f"  maker    {metrics.maker_fill_pct:.0%} of fills")
    for strategy in engine.strategies:
        health = strategy.health()
        skips = sorted(health.get("skips", {}).items(), key=lambda kv: -kv[1])[:4]
        print(f"  {strategy.strategy_id}: intents={health.get('intents', 0)} "
              f"{', '.join(f'{k}={v}' for k, v in skips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
