"""Replay banked market data through the real strategies and risk engines.

This is where simulated trades come from. The backtester is not a separate
simplified model of the platform: it constructs the same ``EventRiskEngine`` the
live process constructs, runs the same strategy classes, and computes features
through the same ``FeaturePipeline``. What differs is the clock, the source of
market data, and that fills are simulated rather than sent.

Usage::

    .venv/Scripts/python.exe scripts/backtest.py
    .venv/Scripts/python.exe scripts/backtest.py --hours 48 --capital 200
    .venv/Scripts/python.exe scripts/backtest.py --strategies mean_reversion,momentum

Widening a limit to make trades appear is the one thing not to do with this. A
run that produces no trades is telling you the strategies declined the data, and
the skip tallies at the end say why.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from kat.common.enums import MarketType
from kat.config.settings import AppSettings, EventRiskSettings
from kat.observability.logging import configure_logging
from kat.simulation.engine import BacktestConfig, BacktestEngine, EventFeed, book_event
from kat.simulation.features import SimFeatureProvider
from kat.strategies.event.expiry_drift import ExpiryDriftStrategy
from kat.strategies.event.mean_reversion import MeanReversionStrategy
from kat.strategies.event.momentum import MomentumStrategy

STRATEGIES = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "expiry_drift": ExpiryDriftStrategy,
}


async def build_feed(
    hours: int, markets: int
) -> tuple[EventFeed, dict[str, int], list[Any]]:
    """Turn stored quotes into a replayable feed.

    Reuses the loader the trainer uses, so a backtest and a training set are
    built from exactly the same reconstruction of the same rows. Two loaders
    would eventually disagree, and the disagreement would look like alpha.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_model import load_histories

    histories = await load_histories(hours, markets)
    events = []
    seq = 0
    for history in histories.values():
        for obs in history._books:  # the feed is built from these
            book = obs.book
            # bids/asks are {price_cc: qty_qc}; the feed wants ordered levels.
            best_bid = book.best_bid_cc
            best_ask = book.best_ask_cc
            if best_bid is None or best_ask is None:
                continue
            bids = [(best_bid, book.bids[best_bid])]
            asks = [(best_ask, book.asks[best_ask])]
            seq += 1
            events.append(
                book_event(
                    history.ticker,
                    info_ts_ms=obs.ts_ms,
                    bids=bids,
                    asks=asks,
                    seq=seq,
                )
            )
    closes = {t: h.close_ts_ms for t, h in histories.items() if h.close_ts_ms}
    return EventFeed(events, name="banked-quotes"), closes, events


def _summarise(result: Any, engine: BacktestEngine) -> None:
    print()
    print("  --- what the run did -------------------------------------------")
    print(f"  intents generated : {engine.intents_generated}")
    print(f"  orders submitted  : {engine.orders_submitted}")
    print(f"  markets too stale : {engine.skipped_stale}")
    if engine.risk_decisions:
        print("  risk verdicts     :")
        for code, count in sorted(engine.risk_decisions.items(), key=lambda kv: -kv[1]):
            print(f"      {count:>6}  {code}")

    print()
    print("  --- why strategies declined ------------------------------------")
    for strategy in engine.strategies:
        health = strategy.health() if hasattr(strategy, "health") else {}
        skips = health.get("skips") or {}
        top = ", ".join(f"{k}={v}" for k, v in list(skips.items())[:4]) or "(never declined)"
        print(f"  {strategy.strategy_id:<26} intents={health.get('intents', 0):<5} {top}")

    trades = getattr(result, "trades", None) or []
    print()
    print(f"  --- {len(trades)} simulated trades ---------------------------------")
    for trade in trades[:12]:
        print(f"    {trade}")
    if len(trades) > 12:
        print(f"    ... and {len(trades) - 12} more")

    for name in ("metrics", "summary", "stats"):
        block = getattr(result, name, None)
        if block:
            print()
            print(f"  --- {name} ---")
            print(f"  {block}")
            break


async def main() -> int:
    parser = argparse.ArgumentParser(description="Replay banked data through the real engines.")
    parser.add_argument("--hours", type=int, default=96)
    parser.add_argument("--markets", type=int, default=150)
    parser.add_argument("--capital", type=float, default=200.0, help="starting capital, USD")
    parser.add_argument(
        "--strategies",
        default="mean_reversion,momentum,expiry_drift",
        help=f"comma separated, from: {', '.join(STRATEGIES)}",
    )
    parser.add_argument(
        "--max-spread-cc",
        type=int,
        default=None,
        help="override the spread limit for this run only. Nothing is persisted; "
             "this exists to answer 'what would it have done on tighter markets', "
             "not to make a bad market look tradeable.",
    )
    args = parser.parse_args()

    configure_logging("WARNING")

    chosen = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in STRATEGIES]
    if unknown:
        print(f"  unknown strategies: {unknown}", file=sys.stderr)
        return 1

    feed, closes, events = await build_feed(args.hours, args.markets)
    if not len(feed):
        print("  no book events to replay.", file=sys.stderr)
        return 1
    print(f"  feed     {len(feed)} book events across {len(closes)} markets")

    strategies = []
    for name in chosen:
        strategy = STRATEGIES[name]()
        if args.max_spread_cc is not None:
            strategy.metadata.params["max_spread_cc"] = args.max_spread_cc
        strategies.append(strategy)

    risk = EventRiskSettings()
    engine = BacktestEngine(
        feed=feed,
        strategies=strategies,
        config=BacktestConfig(
            market_type=MarketType.EVENT,
            initial_capital_micros=int(args.capital * 1_000_000),
        ),
        settings=AppSettings(),
        event_risk_settings=risk,
        # Without this the expiry features are unknown and every strategy that
        # reads them declines every market -- the close times are fetched with
        # the histories and would otherwise be thrown away here.
        feature_provider=SimFeatureProvider(
            events, MarketType.EVENT, close_ts_ms=closes
        ),
    )
    result = engine.run()
    _summarise(result, engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
