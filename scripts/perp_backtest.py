"""Measure the perpetuals domain, which had never been measured at all.

Six structures were tested on Kalshi's event contracts and all six lost, for one
reason that kept recurring: the tolls are enormous relative to anything a small
account can find. A fifty-cent contract pays 175 centicents a side, and the
minimum tick is a whole cent -- on a three-cent contract that single tick is a
third of the notional. No edge measured here survived a round trip.

Perpetuals are a different instrument on the same exchange, and the arithmetic
is not close:

===================  ==========================  ====================
                     event contract              perpetual
===================  ==========================  ====================
fee                  175cc at 50c = 350bp        2bp maker, 4bp taker
tick                 100cc = 3,333bp at 3c       1cc = 0.8bp at $1.26
spread               100-550cc round trip        5-20bp
touch queue          median 124,000 contracts    $49-$100 of notional
===================  ==========================  ====================

Two orders of magnitude, in the direction that decides whether anything works.
So every negative result so far says something about event contracts and nothing
about this.

This runs the perpetual strategies through the same fill simulator that has
already refused five event-contract strategies, over a feed built from Kalshi's
per-minute perpetual candles -- which, unlike the event ones, are dense: 360
candles for 360 minutes, with bid, ask, traded price and volume on each.

What it is careful about
------------------------
Fees are the thing that had to be fixed first. The simulator had a single fee
path and it was the quadratic event-contract one, so anything perpetual would
have been charged about a hundred times too much and rejected on arithmetic
alone. :class:`kat.kalshi.fees.PerpFeeSchedule` is fetched live per market here
rather than assumed.

Queue depth is the thing that killed market making on event contracts, so the
book is seeded with each market's *real* resting size at the touch rather than
an invented constant. It is a snapshot applied across the window, which is the
honest limit of it -- the exchange publishes no historical depth.

Leverage is not a parameter here or anywhere. The perpetual risk engine caps
projected exposure at equity, and this runs through it like anything else.

On repeating a run
------------------
Depth is re-snapshotted on every invocation, and the queue it implies changes
what fills. Two runs of identical parameters returned -7.49% and -17.88%, so a
single figure from this is a draw from a wide distribution and not a
measurement. The sign has been stable across every run; the magnitude has not,
and quoting one to the basis point would be false precision. Fixing that means
banking depth over time rather than sampling it once, which the engine can only
do once perpetual credentials are configured.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

from kat.common.enums import MarketType
from kat.config.settings import AppSettings
from kat.kalshi.auth import KalshiSigner
from kat.kalshi.fees import PerpFeeSchedule
from kat.kalshi.http import KalshiHTTPClient
from kat.kalshi.perps.rest import PerpsREST
from kat.kalshi.ratelimit import RateLimiter
from kat.observability.logging import configure_logging
from kat.simulation.costs import CostModel
from kat.simulation.engine import (
    BacktestConfig,
    BacktestEngine,
    EventFeed,
    book_event,
    ticker_event,
    trade_event,
)
from kat.strategies.perp.funding_carry import FundingCarryStrategy
from kat.strategies.perp.passive_quote import PerpPassiveQuoteStrategy
from kat.strategies.perp.trend import PerpTrendStrategy

STRATEGIES = {
    "funding_carry": FundingCarryStrategy,
    "trend": PerpTrendStrategy,
    "passive_quote": PerpPassiveQuoteStrategy,
}

#: Funding settles every four hours on Kalshi perpetuals -- six times a day,
#: not twenty-four. Annualising an hourly rate overstates the carry fourfold,
#: which is the difference between a headline and a fact.
FUNDING_PERIOD_HOURS = 4

#: Seconds of minute-candles per request. The exchange caps a response near
#: 5,000 candles and returns an empty list rather than an error past it.
CHUNK_SECONDS = 60 * 3600

#: How long a resting quote lives before it is cancelled, matching the live
#: trading loop's own TTL so the two measure the same strategy.
ORDER_TTL_MS = 60_000


def _cc(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value) * 10_000)
    except (TypeError, ValueError):
        return None


def _qc(value: Any) -> int:
    try:
        return round(float(value) * 100)
    except (TypeError, ValueError):
        return 0


async def _client() -> KalshiHTTPClient:
    settings = AppSettings()
    creds = settings.kalshi.active_credentials
    signer = KalshiSigner.from_credentials(creds, label="event")
    return KalshiHTTPClient(
        base_url=settings.kalshi.perp_rest_base_url,
        signer=signer,
        rate_limiter=RateLimiter(),
        is_perp=True,
        timeout_s=settings.kalshi.request_timeout_s,
        connect_timeout_s=settings.kalshi.connect_timeout_s,
    )


async def build_feed(
    rest: PerpsREST, markets: list[Any], *, hours: int, depth: dict[str, int],
    funding: dict[str, float],
) -> EventFeed:
    """One book update, one print and one funding tick per candle."""
    end_ts = int(time.time())
    start_ts = end_ts - hours * 3600
    events: list[Any] = []
    seq = 0
    with_volume = 0

    for market in markets:
        queue_qc = depth.get(market.ticker, 10_000)
        rate = funding.get(market.ticker)
        # Minute candles are capped near 5,000 per request -- about 72 hours --
        # and asking for more returns nothing rather than truncating, which is
        # how a two-week request silently became an empty feed. Paged in chunks
        # under the cap so a longer window is a longer sample and not a
        # different resolution: hour candles reach back thirty days but would
        # hide every intra-hour fill this strategy exists to earn.
        candles = []
        for chunk_end in range(end_ts, start_ts, -CHUNK_SECONDS):
            chunk_start = max(start_ts, chunk_end - CHUNK_SECONDS)
            try:
                candles.extend(await rest.get_candlesticks(
                    market.ticker, start_ts=chunk_start, end_ts=chunk_end,
                    period_interval=1,
                ))
            except Exception as exc:
                print(f"  {market.ticker} {chunk_start}: {type(exc).__name__}",
                      file=sys.stderr)
        if not candles:
            continue

        for candle in candles:
            ts_ms = int(candle.get("end_period_ts", 0)) * 1000
            bid = _cc((candle.get("bid") or {}).get("close"))
            ask = _cc((candle.get("ask") or {}).get("close"))
            if ts_ms <= 0 or bid is None or ask is None or not 0 < bid < ask:
                continue

            seq += 1
            events.append(
                book_event(
                    market.ticker, info_ts_ms=ts_ms,
                    bids=[(bid, queue_qc)], asks=[(ask, queue_qc)], seq=seq,
                )
            )
            seq += 1
            events.append(
                ticker_event(
                    market.ticker, info_ts_ms=ts_ms,
                    mark_price_cc=(bid + ask) // 2,
                    funding_rate=rate, seq=seq,
                )
            )

            volume_qc = _qc(candle.get("volume"))
            if volume_qc <= 0:
                continue
            with_volume += 1
            price = candle.get("price") or {}
            low, high = _cc(price.get("low")), _cc(price.get("high"))
            # Prints at the prices that actually traded. A resting bid is
            # reached when somebody sells down to it; a print at the mid
            # reaches nobody, which is what made the first event-contract feed
            # report twenty-one fills where there should have been a hundred.
            prints = []
            if low is not None and high is not None and low != high:
                half = volume_qc // 2
                prints = [(low, half), (high, volume_qc - half)]
            elif low is not None:
                prints = [(low, volume_qc)]
            for offset, (price_cc, qty_qc) in enumerate(prints):
                if qty_qc <= 0 or price_cc <= 0:
                    continue
                seq += 1
                events.append(
                    trade_event(
                        market.ticker, info_ts_ms=ts_ms + 1 + offset,
                        price_cc=price_cc, qty_qc=qty_qc, seq=seq,
                    )
                )

    events.sort(key=lambda e: (e.info_ts_ms, e.seq))
    print(f"  feed     {len(events):,} events, {with_volume:,} candles with volume, "
          f"{len(markets)} markets")
    return EventFeed(events, name="perp-candles")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--capital", type=float, default=200.0)
    parser.add_argument("--strategy", nargs="*", default=["passive_quote"])
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    configure_logging("WARNING")
    http = await _client()
    rest = PerpsREST(http)
    try:
        markets = [
            m for m in await rest.get_markets(limit=200, max_pages=3)
            if m.bid_cc and m.ask_cc and m.bid_cc < m.ask_cc
        ]
        if args.tickers:
            wanted = set(args.tickers)
            markets = [m for m in markets if m.ticker in wanted]
        if not markets:
            print("no quoted perpetual", file=sys.stderr)
            return 1

        try:
            tiers = await rest.get_fee_tiers()
        except Exception:
            tiers = {}

        depth: dict[str, int] = {}
        funding: dict[str, float] = {}
        for market in markets:
            try:
                book = await rest.get_orderbook(market.ticker, depth=5)
                queue = min(book.best_bid_size_qc, book.best_ask_size_qc)
                if queue > 0:
                    depth[market.ticker] = queue
            except Exception:
                pass
            try:
                estimate = await rest.get_funding_estimate(market.ticker)
                rate = getattr(estimate, "rate", None)
                if rate is None:
                    rate = getattr(estimate, "funding_rate", None)
                if rate is not None:
                    funding[market.ticker] = float(rate)
            except Exception:
                pass

        # Every market's own rates. Assuming one market's discount for another
        # would be inventing a cost, and the fee is the entire reason this
        # instrument is worth testing.
        schedules = {
            m.ticker: PerpFeeSchedule.from_fee_tiers(tiers, m.ticker) for m in markets
        }
        exact = sum(1 for s in schedules.values() if s.exact)
        sample = next(iter(schedules.values()))
        print(f"  markets  {len(markets)} quoted perpetuals")
        print(f"  fees     exact for {exact}/{len(markets)}; "
              f"maker {sample.maker_rate}, taker {sample.taker_rate}")
        if depth:
            sizes = sorted(depth.values())
            print(f"  depth    real for {len(depth)}/{len(markets)}, "
                  f"median {sizes[len(sizes) // 2] / 100:,.0f} contracts at the touch")

        feed = await build_feed(
            rest, markets, hours=args.hours, depth=depth, funding=funding
        )
    finally:
        await http.aclose()

    strategies = [STRATEGIES[name]() for name in args.strategy if name in STRATEGIES]
    if not strategies:
        print(f"unknown strategy; known: {sorted(STRATEGIES)}", file=sys.stderr)
        return 1

    # One schedule for the run. Every Kalshi perpetual currently quotes the same
    # 2bp/4bp pair, so this is exact rather than a compromise -- and it is
    # asserted rather than assumed.
    rates = {(s.maker_rate, s.taker_rate) for s in schedules.values() if s.exact}
    perp_fees = (
        PerpFeeSchedule(maker_rate=r[0], taker_rate=r[1], exact=True)
        if len(rates) == 1 and (r := next(iter(rates)))
        else PerpFeeSchedule()
    )
    if len(rates) > 1:
        print("  NOTE: perpetuals quote differing rates; falling back to the "
              "conservative default for the whole run")

    engine = BacktestEngine(
        feed=feed,
        strategies=strategies,
        config=BacktestConfig(
            market_type=MarketType.PERP,
            initial_capital_micros=int(args.capital * 1_000_000),
            costs=CostModel(
                perp_fees=perp_fees,
                # The live loop cancels a working order after a minute, and a
                # backtest that never cancels is not measuring the same
                # strategy. Left unset, quotes rested for the whole window:
                # unfilled orders accumulated across two weeks until projected
                # exposure reached three quadrillion dollars on a $205 account
                # and the leverage invariant -- correctly -- refused everything
                # after that point. A market maker re-quotes; it does not leave
                # a fortnight of stale bids in the book.
                order_ttl_ms=ORDER_TTL_MS,
            ),
            decision_interval_ms=60_000,
        ),
        settings=AppSettings(),
    )
    result = engine.run()

    trades = getattr(result, "trades", None) or []
    distinct = {
        (t.ticker, t.entry_ms, t.exit_ms, t.qty_qc, t.net_pnl_micros) for t in trades
    }
    m = result.metrics
    print()
    print(f"  fills    {len(engine.fills.fills)}")
    print(f"  trades   {len(trades)} booked, {len(distinct)} distinct")
    print(f"  return   {m.total_return:+.4%}")
    print(f"  sharpe   {m.sharpe:.2f}")
    print(f"  win rate {m.win_rate:.0%}")
    print(f"  fees     ${m.total_fees_micros / 1e6:.4f}")
    print(f"  maker    {m.maker_fill_pct:.0%} of fills")
    if trades:
        import collections as _c
        by_reason = _c.defaultdict(lambda: [0, 0])
        for t in trades:
            r = by_reason[getattr(t, "exit_reason", "?") or "?"]
            r[0] += 1
            r[1] += t.net_pnl_micros
        print()
        print("  exits:")
        for reason, (n, pnl) in sorted(by_reason.items(), key=lambda kv: kv[1][1]):
            print(f"    {reason:<22}{n:>6} trades  ${pnl/1e6:+9.4f}")
        worst = sorted(trades, key=lambda t: t.net_pnl_micros)[:5]
        print("  worst five:")
        for t in worst:
            print(f"    {t.ticker:<14}{t.qty_qc/100:>8.0f}c  ${t.net_pnl_micros/1e6:+9.4f}"
                  f"  {getattr(t,'exit_reason','?')}")
        gross = sum(t.net_pnl_micros for t in trades)
        print(f"  net over all trades: ${gross/1e6:+.4f}")

    for strategy in engine.strategies:
        health = strategy.health()
        skips = sorted(health.get("skips", {}).items(), key=lambda kv: -kv[1])[:4]
        print(f"  {strategy.strategy_id}: intents={health.get('intents', 0)} "
              f"{', '.join(f'{k}={v}' for k, v in skips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
