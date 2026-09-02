"""How accurate would a directional perpetual model have to be, and how close is one?

Eight structures were measured on this platform and all eight lost, seven of
them because Kalshi's event-contract tolls dwarf anything a small account finds.
Perpetuals are not that market, and the difference is worth stating precisely
before any model is fitted:

    median |15-minute move|      23.54 bp
    median round-trip cost       14.21 bp
    windows moving further
      than their own cost        66.7%

On event contracts the cost was the wall. Here it is not: two windows in three
move further than it costs to trade them, and a taker who always guessed the
direction correctly would make 14.12bp a window after fees. So the whole
question reduces to direction, and direction has an exact bar::

    p = 0.5 + cost / (2 * |move|)

At the median that is 80%, which nobody reaches. But the bar falls as the move
grows, so a model does not need to be right about everything -- it needs to be
right about the windows that are large. Those are two separate properties and
nothing guarantees they coincide.

What this measures
------------------
A gradient-boosted classifier on the sign of the forward 15-minute move, from
momentum at four lookbacks, realised volatility, spread and volume. Trained on
the earlier ten days of candles and evaluated once on the four days after them,
never fitted on the test window.

Then the part that matters: expected value per trade, in basis points, with the
round trip already subtracted, sliced by how confident the model is *and* how
volatile the market is.

A correction, and the thing it reveals
--------------------------------------
The numbers below were measured on ten days of fitting and four of testing, and
they do not replicate. Re-run on forty-five days and twelve, the same
top-decile-of-both cell gives 55.6% accuracy rather than 65.1%, against a
break-even bar of 126.3% rather than 72.6%, for an expected value of -108bp
rather than -2.42bp. The short window was not merely noisy, it was flattering.

The reason is worth more than the model was. A bar above 100% means the cost
exceeds the move outright: in that cell the round trip is about 112bp against a
73.5bp move. High volatility on these markets arrives with proportionally wider
spreads, so selecting for windows that move is also selecting for windows that
are expensive, and the two cancel. That is why picking the volatile tail cannot
rescue a directional edge here, and it is a property of the venue rather than of
any model fitted to it.

Both sets of figures are kept below, because the difference between them is the
finding.

The result on the short window, and why it is not a strategy
------------------------------------------------------------
The two properties do coincide. Accuracy rises from 52.3% pooled to 67.9% on
the windows that are both confidently predicted and genuinely moving, and
expected value rises with it from -14.84bp to about zero.

It stops there. The single positive cell is +1.37bp over 302 trades with a
confidence interval of +/-8.20 -- six times the estimate, which is the shape of
noise and not of an edge. This session produced four positives that dissolved
on inspection, every one of them a small sample read too eagerly, and this has
the same signature. The break-even bar on that cell is 77.1% and the model
reaches 67.9%.

Nine points of accuracy is a real distance, not a rounding error. But it is a
margin rather than a wall, which is more than any other structure measured here
managed, and it says exactly what would have to change: better direction, or
cheaper execution.

The ``need@rest`` column, and why it is not the answer
------------------------------------------------------
Cheaper execution is the larger lever, because the spread is the biggest term in
the cost and a rested entry does not pay it. That bar is reported alongside the
taker one, and two buckets clear it -- 65.1% against 63.5% needed, over 1,574
trades, and 67.9% against 65.5% over 302.

It would be a mistake to read that as an edge, for a reason this platform has
already measured twice. The bar assumes a resting entry fills on every signal,
and it does not. A resting bid is reached when the market comes back to it,
which is when the signal was wrong or slow, and it is missed when the price runs
away, which is when the signal was right. Passive entry therefore selects
against exactly the trades the model got right, and nothing in this calculation
accounts for that: the margin is 1.6 points and the bias is plausibly larger.

Settling it needs the fill simulator, which models queue position honestly and
has already refused two passive structures on this platform. That is a
well-posed experiment -- rest the entry, cross the exit, only in the top decile
of confidence and volatility -- and it is the one worth running next. It is not
run here, and this file must not be cited as though it were.
"""

from __future__ import annotations

import argparse
import collections
import math
import pickle
import sys

import numpy as np

#: Forward horizon, in minute-candles.
HORIZON = 15

#: Lookbacks for the momentum features, in minute-candles.
LOOKBACKS = (1, 5, 15, 60)

#: Taker fees for a round trip, in basis points: 4bp in and 4bp out.
ROUND_TRIP_FEE_BPS = 8.0

#: The same round trip with the *entry* rested rather than crossed: 2bp maker
#: in, 4bp taker out, and no spread paid on the way in. The exit still crosses,
#: because a position being closed on a signal cannot wait for a fill.
#: Reported alongside the taker bar because the spread is the largest term in
#: the cost and removing half of it moves the break-even accuracy by more than
#: any amount of model tuning realistically would.
PASSIVE_FEE_BPS = 6.0

#: A mid below this is a broken candle, not a cheap perpetual. Left in, one of
#: them produced a "move" of 2.4e14 basis points and took the mean of every
#: bucket with it.
MIN_MID_CC = 100

#: Moves beyond this are data faults rather than markets.
MAX_MOVE_BPS = 2_000

#: Below this a bucket is not reported: a handful of trades can clear zero by
#: luck, and this analysis exists to not be fooled by that.
MIN_BUCKET = 200


def _series(path: str) -> tuple[dict, dict, str]:
    """Per-ticker minute series of (ts, bid, ask), and per-minute volume."""
    with open(path, "rb") as fh:
        events, _fees, summary = pickle.load(fh)
    prices: dict[str, list] = collections.defaultdict(list)
    volume: dict[str, dict] = collections.defaultdict(dict)
    for event in events:
        if event.kind.value == "book":
            bids = event.payload.get("bids") or []
            asks = event.payload.get("asks") or []
            if bids and asks:
                prices[event.ticker].append(
                    (event.info_ts_ms, bids[0][0], asks[0][0])
                )
        elif event.kind.value == "trade":
            minute = event.info_ts_ms // 60_000
            book = volume[event.ticker]
            book[minute] = book.get(minute, 0) + event.payload.get("qty_qc", 0)
    for rows in prices.values():
        rows.sort()
    return prices, volume, summary


def build(path: str) -> tuple[np.ndarray, np.ndarray, list, str]:
    """Features, labels and (|move|, cost) per window."""
    prices, volume, summary = _series(path)
    X: list[list[float]] = []
    y: list[int] = []
    meta: list[tuple[float, float, str]] = []
    longest = max(LOOKBACKS)

    for ticker, rows in prices.items():
        if len(rows) < longest * 2:
            continue
        mids = np.array([(b + a) / 2 for _, b, a in rows], dtype=float)
        spreads = np.array([a - b for _, b, a in rows], dtype=float)
        stamps = [t for t, _, _ in rows]

        for i in range(longest, len(rows) - HORIZON):
            mid = mids[i]
            forward = mids[i + HORIZON]
            if mid < MIN_MID_CC or forward < MIN_MID_CC:
                continue
            # Index arithmetic is not time arithmetic: rows are sparse wherever
            # a market went quiet, so the horizon is checked in seconds.
            elapsed = stamps[i + HORIZON] - stamps[i]
            if not (600_000 <= elapsed <= 1_800_000):
                continue
            move_bps = (forward - mid) / mid * 1e4
            if abs(move_bps) > MAX_MOVE_BPS:
                continue

            window = mids[i - longest:i + 1]
            steps = np.diff(window) / window[:-1] * 1e4
            spread_bps = spreads[i] / mid * 1e4
            traded = volume[ticker].get(stamps[i] // 60_000, 0) / 100.0

            X.append([
                *((mid - mids[i - k]) / mid * 1e4 for k in LOOKBACKS),
                float(np.std(steps)) if steps.size > 1 else 0.0,
                spread_bps,
                math.log1p(traded),
            ])
            y.append(1 if move_bps > 0 else 0)
            meta.append((abs(move_bps), spread_bps + ROUND_TRIP_FEE_BPS, ticker))

    return np.array(X), np.array(y), meta, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", help="cached feed from perp_backtest.py --cache")
    parser.add_argument("test", help="a later, untouched cached feed")
    args = parser.parse_args()

    X_train, y_train, _, summary_train = build(args.train)
    X_test, y_test, meta_test, summary_test = build(args.test)
    if len(y_train) < 1_000 or len(y_test) < 1_000:
        print("not enough windows; widen the cached feeds", file=sys.stderr)
        return 1

    print(f"  train {summary_train}: {len(y_train):,} windows, {y_train.mean():.1%} up")
    print(f"  test  {summary_test}: {len(y_test):,} windows, {y_test.mean():.1%} up\n")

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.8, verbose=-1,
    )
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    print(f"  out-of-sample AUC {roc_auc_score(y_test, probability):.4f} "
          f"against 0.5000 for a coin\n")

    move = np.array([m for m, _, _ in meta_test])
    cost = np.array([c for _, c, _ in meta_test])
    right = (probability > 0.5) == (y_test == 1)
    confidence = np.abs(probability - 0.5)
    volatility = X_test[:, len(LOOKBACKS)]

    def report(mask: np.ndarray, label: str) -> None:
        if mask.sum() < MIN_BUCKET:
            return
        # Win the move, lose the move, pay the round trip either way.
        pnl = np.where(right[mask], move[mask], -move[mask]) - cost[mask]
        mean = pnl.mean()
        half = 1.96 * pnl.std() / math.sqrt(len(pnl))
        # The bar this bucket actually had to clear.
        mean_move = move[mask].mean()
        needed = 0.5 + cost[mask].mean() / (2 * mean_move)
        # What the bar becomes if the entry rests instead of crossing: the
        # spread leaves the cost entirely and the entry fee halves.
        passive_cost = (cost[mask] - ROUND_TRIP_FEE_BPS) / 2 + PASSIVE_FEE_BPS
        needed_passive = 0.5 + passive_cost.mean() / (2 * mean_move)
        flag = "  <-- clears zero" if mean - half > 0 else ""
        if right[mask].mean() > needed_passive:
            flag = "  <-- clears the passive bar"
        print(f"  {label:>24}{mask.sum():>9,}{right[mask].mean():>10.1%}"
              f"{needed:>10.1%}{needed_passive:>10.1%}{mean_move:>9.1f}"
              f"{mean:>9.2f}{f'+/-{half:.2f}':>10}{flag}")

    header = (f"  {'selection':>24}{'trades':>9}{'accuracy':>10}"
              f"{'need@take':>10}{'need@rest':>10}{'|move|':>9}"
              f"{'EV bp':>9}{'95% CI':>10}")
    print(header)
    for q in (0.0, 0.5, 0.9, 0.95, 0.99):
        report(confidence >= np.quantile(confidence, q), f"confidence top {100*(1-q):.0f}%")
    print()
    print(header)
    for cq in (0.5, 0.9):
        for vq in (0.5, 0.9, 0.99):
            mask = (
                (confidence >= np.quantile(confidence, cq))
                & (volatility >= np.quantile(volatility, vq))
            )
            report(mask, f"conf {100*(1-cq):.0f}% x vol {100*(1-vq):.0f}%")

    print("\n  EV already nets the round trip; 'needed' is the accuracy that")
    print("  bucket had to reach to break even. A positive cell over a few")
    print("  hundred trades with an interval several times its own size is the")
    print("  shape of noise, and this project has produced four of those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
