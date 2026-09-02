"""How good is the market already? The number every model has to beat.

A model that predicts settlement is not competing with a coin, it is competing
with the price. Kalshi quotes a probability continuously and for free, so a
forecast only has value where it is *better* than that quote -- and a test AUC
reported on its own cannot show whether it is. The first settlement model here
scored 0.69 to 0.71 and looked like signal, because 0.5 is the benchmark that
comes to mind. The price scores 0.7579 over the same markets. The model was
worse than doing nothing, and nothing in its own metrics said so.

This measures the benchmark directly: take the mid at some point before close,
call it the market's probability, and score it against what actually happened.
No model, no features, no fitting -- just the quote and the outcome.

Read it as a floor, not a target. Beating it on AUC is necessary and not
sufficient, because a forecast also has to beat it by more than the round trip
costs, which on this venue runs from 136 to 550 centicents.

The rise toward close is the part worth internalising: the market is at 0.74 six
hours out and 0.84 in the final hour. Late accuracy is when a forecast is least
likely to add anything, and it is exactly when a strategy waiting for
confirmation would want to trade.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from typing import Any

import numpy as np

from kat.config.settings import AppSettings

#: Buckets of time-before-close, in hours. The market's edge over a naive
#: forecast is not constant across a market's life, and one number hides that.
WINDOWS: tuple[tuple[float, float], ...] = (
    (0, 1), (1, 3), (3, 6), (6, 12), (12, 24),
)

#: Below this an AUC is sampling noise wearing a decimal point.
MIN_OBSERVATIONS = 500


def _score(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    return float(roc_auc_score(y, p)), float(brier_score_loss(y, p))


async def _rows(conn: Any, *, hours: int, family: str | None) -> list[Any]:
    return await conn.fetch(
        """
        select split_part(s.ticker, '-', 1) as fam,
               s.result,
               q.bid_cc, q.ask_cc,
               extract(epoch from (s.settled_ts - q.ts)) / 3600.0 as hrs
        from settlements s
        join quotes q on q.ticker = s.ticker
        where q.bid_cc is not null and q.ask_cc is not null
          and q.bid_cc < q.ask_cc
          and s.result in ('yes', 'no')
          and q.ts < s.settled_ts
          and q.ts > s.settled_ts - make_interval(hours => $1)
          and ($2::text is null or split_part(s.ticker, '-', 1) = $2)
        """,
        hours, family,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24,
                        help="how far before close to look")
    parser.add_argument("--family", default=None,
                        help="restrict to one series family, e.g. KXMLBGAME")
    parser.add_argument("--by-family", action="store_true",
                        help="also break the benchmark out per series family")
    args = parser.parse_args()

    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    try:
        rows = await _rows(conn, hours=args.hours, family=args.family)
    finally:
        await conn.close()

    if not rows:
        print("no settled market has a quote history yet; "
              "run scripts/backfill_settled.py first")
        return 1

    y = np.fromiter((r["result"] == "yes" for r in rows), dtype=int, count=len(rows))
    mid = np.fromiter(
        ((r["bid_cc"] + r["ask_cc"]) / 20_000.0 for r in rows),
        dtype=float, count=len(rows),
    )
    hrs = np.fromiter((float(r["hrs"]) for r in rows), dtype=float, count=len(rows))

    print(f"\n  {len(y):,} quoted minutes across settled markets, "
          f"{y.mean():.1%} settled YES\n")
    print(f"  {'before close':>14}{'observations':>14}{'AUC':>9}{'Brier':>9}")
    for lo, hi in WINDOWS:
        mask = (hrs >= lo) & (hrs < hi)
        if mask.sum() < MIN_OBSERVATIONS or len(np.unique(y[mask])) < 2:
            continue
        auc, brier = _score(y[mask], mid[mask])
        print(f"  {f'{lo:g}-{hi:g}h':>14}{mask.sum():>14,}{auc:>9.4f}{brier:>9.4f}")
    auc, brier = _score(y, mid)
    print(f"  {'overall':>14}{len(y):>14,}{auc:>9.4f}{brier:>9.4f}")

    if args.by_family:
        print(f"\n  {'family':<18}{'observations':>14}{'yes':>7}{'AUC':>9}{'Brier':>9}")
        families = np.array([r["fam"] for r in rows])
        for fam in sorted(set(families.tolist())):
            mask = families == fam
            if mask.sum() < MIN_OBSERVATIONS or len(np.unique(y[mask])) < 2:
                continue
            auc, brier = _score(y[mask], mid[mask])
            print(f"  {fam:<18}{mask.sum():>14,}{y[mask].mean():>6.0%}"
                  f"{auc:>9.4f}{brier:>9.4f}")
        # An AUC far below 0.5 is not an inverted edge waiting to be traded, it
        # is a handful of markets: the one family that showed 0.26 had three
        # YES outcomes in twenty-four. Check the count before believing the sign.
        print("\n  An AUC below 0.5 means too few outcomes, not a free trade.")

    print("\n  A settlement model is only useful above these numbers, "
          "and only by more than the round trip costs.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
