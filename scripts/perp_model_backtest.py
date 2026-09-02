"""The experiment perp_direction.py specified and could not settle.

That analysis established the arithmetic. On Kalshi perpetuals the toll is not
the wall -- two windows in three move further than they cost -- so everything
rests on direction, and direction has an exact bar. Crossing both legs the bar
is 72.6% and the model reaches 65.1%. Resting the *entry* removes the spread,
the largest term in the cost, and drops the bar to 63.5%, which the model
clears.

But that is a spreadsheet, and it assumes a resting entry fills whenever the
model fires. It does not. A resting bid is reached when the market comes back to
it -- when the forecast was wrong or slow -- and missed when the price runs,
which is when the forecast was right. Passive entry selects against the trades
worth having, and no amount of arithmetic says by how much.

This runs it properly. The model is fitted on the earlier window and never sees
the later one; its predictions are injected into a backtest over the later
window, and every fill is decided by :mod:`kat.simulation.fills`, which assumes
we sit behind all the size resting at our price and fills only when the market
trades through. That simulator has already refused two passive structures here,
which is the reason to trust it when it does not.

Two things that would quietly invalidate this, and are therefore not done
-------------------------------------------------------------------------
The confidence and volatility gates are quantiles taken from the *training*
window. Taking them from the test window would choose the threshold using the
outcomes it is evaluated against, which is lookahead wearing the clothes of a
hyperparameter.

The prediction is stamped at the close of the minute it was computed from and
not before, so the engine's information-time check can reject anything that
would have been unknowable. A forecast dated a minute early is a forecast of
the past.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from typing import Any

import numpy as np

sys.path.insert(0, "scripts")

from perp_direction import HORIZON, LOOKBACKS, build

from kat.common.enums import MarketType
from kat.config.settings import AppSettings
from kat.simulation.costs import CostModel
from kat.simulation.engine import (
    BacktestConfig,
    BacktestEngine,
    EventFeed,
    ModelPrediction,
)
from kat.strategies.perp.model_quote import TARGET, PerpModelQuoteStrategy

#: How long a prediction stays usable. The model forecasts a fifteen-minute
#: move, so a prediction older than a minute is describing a book that has moved
#: on; the strategy would be acting on a forecast made for a different price.
PREDICTION_TTL_MS = 90_000


def train_model(X: np.ndarray, y: np.ndarray) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.8, verbose=-1,
    )
    model.fit(X, y)
    return model


def index_predictions(
    model: Any, path: str, *, conf_floor: float, vol_floor: float,
) -> tuple[dict[tuple[str, int], ModelPrediction], int, int]:
    """Predictions for every qualifying window, keyed by (ticker, minute).

    Rebuilt from the same feed the backtest replays, so the timestamps line up
    exactly with the book events the strategy will see.
    """
    from perp_direction import _series

    prices, volume, _summary = _series(path)
    longest = max(LOOKBACKS)
    rows: list[tuple[str, int, list[float]]] = []

    for ticker, series in prices.items():
        if len(series) < longest * 2:
            continue
        mids = np.array([(b + a) / 2 for _, b, a in series], dtype=float)
        spreads = np.array([a - b for _, b, a in series], dtype=float)
        stamps = [t for t, _, _ in series]
        for i in range(longest, len(series) - HORIZON):
            mid = mids[i]
            if mid < 100:
                continue
            window = mids[i - longest:i + 1]
            steps = np.diff(window) / window[:-1] * 1e4
            rows.append((
                ticker, stamps[i],
                [
                    *((mid - mids[i - k]) / mid * 1e4 for k in LOOKBACKS),
                    float(np.std(steps)) if steps.size > 1 else 0.0,
                    spreads[i] / mid * 1e4,
                    math.log1p(volume[ticker].get(stamps[i] // 60_000, 0) / 100.0),
                ],
            ))

    if not rows:
        return {}, 0, 0
    features = np.array([r[2] for r in rows])
    probability = model.predict_proba(features)[:, 1]
    volatility = features[:, len(LOOKBACKS)]

    out: dict[tuple[str, int], ModelPrediction] = {}
    kept = 0
    for (ticker, stamp, _f), p, v in zip(rows, probability, volatility, strict=True):
        # Both gates, on thresholds learned from the training window.
        if abs(p - 0.5) < conf_floor or v < vol_floor:
            continue
        kept += 1
        out[(ticker, stamp // 60_000)] = ModelPrediction(
            probability=float(p),
            # Knowable at the close of the minute it was computed from, and not
            # a millisecond earlier.
            info_ts_ms=stamp,
            available_ms=stamp,
            model_id="perp-direction-lgbm",
            model_version="1.0.0",
            target=TARGET,
        )
    return out, kept, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", help="cached feed to fit on")
    parser.add_argument("test", help="a later, untouched cached feed")
    parser.add_argument("--capital", type=float, default=200.0)
    parser.add_argument("--conf-quantile", type=float, default=0.90)
    parser.add_argument("--vol-quantile", type=float, default=0.90)
    parser.add_argument("--param", nargs="*", default=[])
    args = parser.parse_args()

    X_train, y_train, _, summary_train = build(args.train)
    if len(y_train) < 1_000:
        print("not enough training windows", file=sys.stderr)
        return 1
    model = train_model(X_train, y_train)

    # The gates, learned where the outcomes are already spent.
    p_train = model.predict_proba(X_train)[:, 1]
    conf_floor = float(np.quantile(np.abs(p_train - 0.5), args.conf_quantile))
    vol_floor = float(np.quantile(X_train[:, len(LOOKBACKS)], args.vol_quantile))
    print(f"  train    {summary_train}: {len(y_train):,} windows")
    print(f"  gates    confidence >= {conf_floor:.4f} (q{args.conf_quantile:g}), "
          f"volatility >= {vol_floor:.2f}bp (q{args.vol_quantile:g}), "
          f"from the TRAIN window")

    predictions, kept, total = index_predictions(
        model, args.test, conf_floor=conf_floor, vol_floor=vol_floor
    )
    print(f"  signals  {kept:,} of {total:,} test windows qualify "
          f"({100 * kept / max(total, 1):.1f}%)")

    with open(args.test, "rb") as fh:
        events, perp_fees, summary_test = pickle.load(fh)
    print(f"  test     {summary_test}: {len(events):,} events")

    def provider(ticker: str, now_ms: int) -> ModelPrediction | None:
        prediction = predictions.get((ticker, now_ms // 60_000))
        if prediction is None or now_ms - prediction.available_ms > PREDICTION_TTL_MS:
            return None
        return prediction

    overrides: dict[str, Any] = {}
    for item in args.param:
        name, _, raw = item.partition("=")
        try:
            overrides[name] = int(raw)
        except ValueError:
            overrides[name] = float(raw)
    if overrides:
        print(f"  params   {overrides}")

    engine = BacktestEngine(
        feed=EventFeed(events, name="perp-candles"),
        strategies=[PerpModelQuoteStrategy(overrides or None)],
        config=BacktestConfig(
            market_type=MarketType.PERP,
            initial_capital_micros=int(args.capital * 1_000_000),
            costs=CostModel(perp_fees=perp_fees, order_ttl_ms=60_000),
            decision_interval_ms=60_000,
        ),
        settings=AppSettings(),
        model_provider=provider,
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
        pnl = np.array([t.net_pnl_micros for t in trades], dtype=float)
        mean = pnl.mean()
        half = 1.96 * pnl.std() / math.sqrt(len(pnl))
        print(f"  per trade ${mean / 1e6:+.4f} +/- {half / 1e6:.4f}")
        verdict = (
            "POSITIVE" if mean - half > 0
            else "negative" if mean + half < 0
            else "indistinguishable from zero"
        )
        print(f"  verdict  {verdict}")

    # The number that decides whether passive entry works. The model's
    # accuracy is measured over every window it fires on; a resting order is
    # only filled on the subset where the market came back to our price. If
    # those two differ, the gap is adverse selection, measured rather than
    # argued -- and it is the whole reason the spreadsheet bar was optimistic.
    if trades:
        directional = [
            t for t in trades
            if getattr(t, "entry_price_cc", 0) and getattr(t, "exit_price_cc", 0)
        ]
        if directional:
            right = sum(
                1 for t in directional
                if (t.exit_price_cc > t.entry_price_cc) == (t.qty_qc > 0)
            )
            filled_acc = right / len(directional)
            print(f"  direction on FILLED trades: {filled_acc:.1%} "
                  f"over {len(directional):,}")
            print("  the model's own accuracy on the windows it fired on is")
            print("  printed by perp_direction.py; the gap between that and this")
            print("  is what a resting entry selects away.")

    for strategy in engine.strategies:
        health = strategy.health()
        skips = sorted(health.get("skips", {}).items(), key=lambda kv: -kv[1])[:5]
        print(f"  {strategy.strategy_id}: intents={health.get('intents', 0)} "
              f"{', '.join(f'{k}={v}' for k, v in skips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
