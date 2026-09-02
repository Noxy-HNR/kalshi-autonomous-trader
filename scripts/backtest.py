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
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kat.common.enums import MarketType
from kat.config.settings import AppSettings, EventRiskSettings
from kat.observability.logging import configure_logging
from kat.simulation.engine import (
    BacktestConfig,
    BacktestEngine,
    EventFeed,
    ModelPrediction,
    book_event,
    trade_event,
)
from kat.simulation.features import SimFeatureProvider
from kat.strategies.event.expiry_drift import ExpiryDriftStrategy
from kat.strategies.event.mean_reversion import MeanReversionStrategy
from kat.strategies.event.model_drift import ModelDriftStrategy
from kat.strategies.event.momentum import MomentumStrategy
from kat.strategies.event.passive_quote import PassiveQuoteStrategy

STRATEGIES = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "expiry_drift": ExpiryDriftStrategy,
    "model_drift": ModelDriftStrategy,
    "passive_quote": PassiveQuoteStrategy,
}


class DeployedModelProvider:
    """Scores the deployed model at each decision time, as the live path does.

    The point of a backtest is to answer in minutes what live paper trading
    answers in days, and a backtest that cannot serve the model can only measure
    the strategies that ignore it. Features come from the same
    :class:`SimFeatureProvider` the strategies read, so a probability here and a
    probability in production are computed from the same numbers.

    ``info_ts_ms`` is the feature vector's own timestamp rather than the decision
    time, so the engine's leakage check is scoring the information the model
    actually used.
    """

    def __init__(self, model: Any, features: Any) -> None:
        self.model = model
        self.features = features
        self.scored = 0
        self.no_features = 0

    def __call__(self, ticker: str, now_ms: int) -> ModelPrediction | None:
        import numpy as np

        vector = self.features(ticker, now_ms)
        if vector is None or not getattr(vector, "values", None):
            self.no_features += 1
            return None
        row = np.asarray([vector.ordered(self.model.feature_names)], dtype=float)
        if not np.isfinite(row).all():
            self.no_features += 1
            return None
        raw = self.model.estimator.predict_proba(row)[:, 1]
        if self.model.calibrator is not None:
            raw = np.clip(self.model.calibrator.transform(raw), 0.0, 1.0)
        self.scored += 1
        return ModelPrediction(
            probability=float(raw[0]),
            info_ts_ms=int(vector.as_of_ms),
            available_ms=int(vector.as_of_ms),
            model_id=self.model.model_id,
            model_version=self.model.version,
            target=self.model.target,
            base_rate=self.model.base_rate,
        )


async def load_deployed_model() -> Any:
    """Whatever the registry currently has deployed for event contracts."""
    from kat.common.enums import MarketType as _MT
    from kat.db.engine import get_database
    from kat.models.registry import ModelRegistry

    db = get_database()
    await db.connect()
    try:
        return await ModelRegistry(db).deployed(_MT.EVENT)
    finally:
        await db.disconnect()


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
    # Prints, not just books. A resting order fills when the market trades
    # *through* it, so a feed of book snapshots alone can never fill one --
    # which is not a fact about the market, it is a hole in the measurement.
    # Without these, 421 resting quotes over a week produced zero fills and
    # looked like a finding about queue position.
    seq = await _append_trade_prints(events, set(histories), hours, seq)

    events.sort(key=lambda e: (e.info_ts_ms, e.seq))
    closes = {t: h.close_ts_ms for t, h in histories.items() if h.close_ts_ms}
    return EventFeed(events, name="banked-quotes"), closes, events


async def _append_trade_prints(
    events: list[Any], tickers: set[str], hours: int, seq: int
) -> int:
    """Add the banked prints for these markets to the feed."""
    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    since = datetime.now(UTC) - timedelta(hours=hours)
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            select ticker, ts, price_cc, count_qc
            from trade_prints
            where market_type = 'event' and ts >= $1 and ticker = any($2::text[])
              and price_cc is not null and count_qc > 0
            order by ts
            """,
            since, sorted(tickers),
        )
    finally:
        await conn.close()

    for row in rows:
        seq += 1
        events.append(
            trade_event(
                row["ticker"],
                info_ts_ms=int(row["ts"].timestamp() * 1000),
                price_cc=int(row["price_cc"]),
                qty_qc=int(row["count_qc"]),
                seq=seq,
            )
        )
    print(f"  prints   {len(rows)} trades across {len(tickers)} markets")
    return seq


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
    # Distinct as well as total. A ledger that books the same close more than
    # once inflates the trade count, tightens every confidence interval around
    # a repeated number, and makes a handful of losses look like a hundred and
    # forty-four -- which is exactly what a win rate of precisely zero with an
    # expectancy interval of six hundredths of a cent turned out to be.
    distinct = {
        (t.ticker, t.entry_ms, t.exit_ms, t.qty_qc, t.net_pnl_micros) for t in trades
    }
    duplicates = len(trades) - len(distinct)
    suffix = f" ({len(distinct)} distinct, {duplicates} duplicated)" if duplicates else ""
    print(
        f"  --- {len(trades)} simulated trades{suffix} "
        "---------------------------------"
    )
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
    parser.add_argument(
        "--min-edge-cc",
        type=int,
        default=None,
        help="override the risk engine's edge bar FOR THIS RUN, to measure what "
             "trades below it would have done. Never a reason to change the "
             "shipped default, which exists because those trades lost money.",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="run without the deployed model, to see what the rest does alone",
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

    features = SimFeatureProvider(events, MarketType.EVENT, close_ts_ms=closes)
    model_provider = None
    if not args.no_model:
        model = await load_deployed_model()
        if model is None:
            print("  no model deployed; strategies that need one will stand aside")
        else:
            print(f"  model    {model.key} target={model.target}")
            model_provider = DeployedModelProvider(model, features)

    # An experiment knob, not a production setting. The shipped min_edge_cc is
    # 200 and stays there; this exists so a run can ask "what would these trades
    # have done if the bar had let them through" and get a number instead of an
    # opinion. Answering that is the whole point of a backtester.
    risk = EventRiskSettings()
    if args.min_edge_cc is not None:
        risk = EventRiskSettings(min_edge_cc=args.min_edge_cc)
        print(f"  EXPERIMENT: min_edge_cc lowered to {args.min_edge_cc} for this run only")
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
        feature_provider=features,
        model_provider=model_provider,
    )
    result = engine.run()
    _summarise(result, engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
