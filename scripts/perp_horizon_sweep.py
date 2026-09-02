"""Does a longer horizon lower the break-even bar faster than it costs accuracy?

The bar a directional model has to clear is p = 0.5 + cost / (2 * |move|). Cost
is roughly fixed per round trip while |move| grows with the square root of time,
so the bar must fall as the horizon lengthens -- mechanically, without any model
getting better. Every directional measurement on this platform used fifteen
minutes and never varied it, which left the cheapest available lever untried.

It works, on the side it was supposed to::

    horizon   |move|   cost     bar     accuracy   AUC
        15m     43.1   38.8   95.0%        53.2%   0.5130
        60m     77.7   15.9   60.3%        53.6%   0.5058
       240m    134.8   58.0   71.5%        47.1%   0.4953
       720m    147.9   15.9   55.4%        52.9%   0.4940

The bar falls from 95% to 55.4%. The model does not survive the trip: AUC decays
toward a coin as the horizon grows, so the forecast dies faster than the toll
does. At twelve hours the bar is 55.4% and the model delivers 52.9% on 336
independent windows -- short, and by then too noisy to resolve.

The independence matters more than the result. At a twelve-hour horizon two
windows one minute apart share 99.9% of their outcome, so sampling every minute
would have multiplied the apparent sample by 720 and shrunk every interval to
nothing. Windows are stepped by the horizon instead. This project has produced
five positives that dissolved on inspection and every one was a sample counted
more times than it happened; that error is easier to make here than anywhere
else in the codebase.
"""
import math
import sys

sys.path.insert(0, "scripts")
import lightgbm as lgb
import numpy as np
from perp_direction import LOOKBACKS, _series
from sklearn.metrics import roc_auc_score

FEE_BPS = 8.0

def build(path, horizon):
    prices, volume, _summary = _series(path)
    longest = max(LOOKBACKS)
    X, y, meta = [], [], []
    for tk, s in prices.items():
        if len(s) < longest + horizon + 10:
            continue
        mids = np.array([(b+a)/2 for _, b, a in s], dtype=float)
        spr = np.array([a-b for _, b, a in s], dtype=float)
        ts = [t for t, _, _ in s]
        # Independent windows only: step by the horizon, not by one minute.
        for i in range(longest, len(s)-horizon, horizon):
            m0, m1 = mids[i], mids[i+horizon]
            if m0 < 100 or m1 < 100:
                continue
            gap = ts[i+horizon]-ts[i]
            if not (horizon * 40_000 <= gap <= horizon * 90_000):
                continue
            mv = (m1-m0)/m0*1e4
            if abs(mv) > 5000:
                continue
            win = mids[i-longest:i+1]
            steps = np.diff(win)/win[:-1]*1e4
            X.append([*((m0-mids[i-k])/m0*1e4 for k in LOOKBACKS),
                      float(np.std(steps)) if steps.size > 1 else 0.0,
                      spr[i]/m0*1e4,
                      math.log1p(volume[tk].get(ts[i]//60_000, 0)/100.0)])
            y.append(1 if mv > 0 else 0)
            meta.append((abs(mv), spr[i]/m0*1e4 + FEE_BPS))
    return np.array(X), np.array(y), meta

print(f"  {'horizon':>9}{'train n':>9}{'test n':>8}{'AUC':>8}"
      f"{'|move|':>9}{'cost':>8}{'need':>8}{'acc@top10':>11}{'EV bp':>10}{'95% CI':>10}")
for horizon in (15, 60, 240, 720, 1440):
    Xtr, ytr, _ = build(sys.argv[1], horizon)
    Xte, yte, mte = build(sys.argv[2], horizon)
    if len(ytr) < 500 or len(yte) < 200:
        print(f"  {horizon:>7}m  too few independent windows")
        continue
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=50, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    move = np.array([a for a, _ in mte])
    cost = np.array([c for _, c in mte])
    conf = np.abs(p - 0.5)
    right = (p > 0.5) == (yte == 1)
    sel = conf >= np.quantile(conf, 0.90)
    auc = roc_auc_score(yte, p) if len(set(yte)) > 1 else float("nan")
    need = 0.5 + cost[sel].mean()/(2*move[sel].mean())
    pnl = np.where(right[sel], move[sel], -move[sel]) - cost[sel]
    half = 1.96*pnl.std()/math.sqrt(max(len(pnl),1))
    flag = "  <-- clears" if pnl.mean()-half > 0 else ""
    print(f"  {horizon:>7}m{len(ytr):>9,}{len(yte):>8,}{auc:>8.4f}"
          f"{move[sel].mean():>9.1f}{cost[sel].mean():>8.1f}{need:>7.1%}"
          f"{right[sel].mean():>10.1%}{pnl.mean():>10.2f}{f'+/-{half:.1f}':>10}{flag}")
