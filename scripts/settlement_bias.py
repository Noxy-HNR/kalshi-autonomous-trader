"""Is the market's own price systematically wrong, and by enough to trade?

Every other structure measured on this venue needs a forecast that beats the
quote, and none of them did: the price scores 0.7764 AUC against settlement and
the best model trained here scored 0.71. This asks a different question, one
that needs no forecast at all. Not *which* contract will settle YES, but whether
contracts priced at 80 cents settle YES *more often than 80% of the time*.

That is the favourite-longshot bias, the most durable finding in the literature
on betting and prediction markets: longshots are overpriced and favourites are
underpriced, persistently, because the demand is for lottery tickets. If it is
present here it is worth money without predicting anything, and it is worth more
here than at a bookmaker because a contract held to settlement pays no exit
spread and no exit fee -- half the round trip that killed every other structure
simply does not occur.

Counting it correctly is the entire difficulty
----------------------------------------------
A market quoted at 85 cents for a thousand minutes is *one* observation of
whether things priced at 85 cents happen, not a thousand. The outcome is drawn
once. Pooling the minutes multiplies the apparent sample by three orders of
magnitude while adding no information, and the confidence interval it produces
is fiction -- on this dataset it shrank an interval of +/-596cc to +/-19cc and
turned "we cannot tell" into "certain".

So this samples each market exactly once, at a fixed time before close, and the
interval it reports is a binomial interval on the number of *markets*. That is
usually a disappointing number. It is the true one.

Reading the result
------------------
``buy YES @ask`` is what the trade actually nets per contract: the settlement
value, less the ask crossed to get in, less the fee on entry. Nothing is
subtracted for exit because there is no exit. A band is only tradeable when the
low end of its interval clears zero -- a positive point estimate inside a wide
interval is the shape of the bias, not evidence of it.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import re
from collections import defaultdict
from itertools import pairwise
from typing import Any

from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.config.settings import AppSettings
from kat.kalshi.fees import BASE_COEFFICIENT

#: Price bands, in centicents. Ten-cent buckets are wide enough to hold enough
#: markets to say anything and narrow enough that the bias is not averaged away.
BANDS: tuple[int, ...] = tuple(range(0, CC_PER_DOLLAR + 1, 1_000))

#: Below this a band is reported but never called tradeable, however the
#: interval falls: a handful of markets can clear zero by luck.
MIN_MARKETS = 30


def fee_cc(price_cc: float) -> float:
    """Kalshi's quadratic fee at a price, in centicents per contract."""
    p = price_cc / CC_PER_DOLLAR
    return float(BASE_COEFFICIENT) * p * (1.0 - p) * CC_PER_DOLLAR


async def sample(conn: Any, *, minutes_before: int, window: int,
                 family: str | None) -> list[Any]:
    """One quote per settled market, the last one before the cutoff."""
    return await conn.fetch(
        """
        select distinct on (s.ticker)
               s.ticker,
               split_part(s.ticker, '-', 1) as fam,
               s.result, q.bid_cc, q.ask_cc
        from settlements s
        join quotes q on q.ticker = s.ticker
        where q.bid_cc is not null and q.ask_cc is not null
          and q.bid_cc < q.ask_cc
          and s.result in ('yes', 'no')
          and q.ts <  s.settled_ts - make_interval(mins => $1)
          and q.ts >  s.settled_ts - make_interval(mins => $2)
          and ($3::text is null or split_part(s.ticker, '-', 1) = $3)
        order by s.ticker, q.ts desc
        """,
        minutes_before, minutes_before + window, family,
    )


def report(rows: list[Any], *, label: str) -> None:
    buckets: dict[tuple[int, int], list[tuple[bool, float, float]]] = defaultdict(list)
    for row in rows:
        mid = (row["bid_cc"] + row["ask_cc"]) / 2.0
        for lo, hi in pairwise(BANDS):
            if lo <= mid < hi:
                buckets[(lo, hi)].append(
                    (row["result"] == "yes", mid, float(row["ask_cc"]))
                )
                break

    print(f"\n  {label}: {len(rows):,} markets, one quote each\n")
    print(f"  {'band':>10}{'markets':>9}{'mid':>7}{'settled YES':>13}"
          f"{'buy YES @ask':>14}{'95% CI':>12}{'verdict':>11}")
    tradeable = []
    for lo, hi in pairwise(BANDS):
        values = buckets.get((lo, hi)) or []
        if len(values) < 10:
            continue
        n = len(values)
        wins = sum(1 for hit, _, _ in values if hit)
        rate = wins / n
        mid = sum(m for _, m, _ in values) / n
        ask = sum(a for _, _, a in values) / n
        # The uncertainty is in the settlement rate, and it is a rate over
        # markets. Everything else here is measured, not estimated.
        half = 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / n) * CC_PER_DOLLAR
        ev = rate * CC_PER_DOLLAR - ask - fee_cc(ask)
        if n < MIN_MARKETS:
            verdict = "too few"
        elif ev - half > 0:
            verdict = "TRADEABLE"
            tradeable.append((lo, hi, ev, half, n))
        elif ev + half < 0:
            verdict = "negative"
        else:
            verdict = "unclear"
        print(f"  {f'{lo // 100}-{hi // 100}c':>10}{n:>9}{mid:>7.0f}"
              f"{rate * CC_PER_DOLLAR:>13.0f}{ev:>14.0f}{f'+/-{half:.0f}':>12}"
              f"{verdict:>11}")

    if tradeable:
        print("\n  Bands whose interval clears zero:")
        for lo, hi, ev, half, n in tradeable:
            print(f"    {lo // 100}-{hi // 100}c  +{ev:.0f}cc per contract "
                  f"(+/-{half:.0f}) over {n} markets")
    else:
        print("\n  No band's interval clears zero. The point estimates may still")
        print("  show the bias's shape -- rising with price -- but shape is not")
        print("  evidence, and the fix is more settled markets, not more minutes.")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes-before", type=int, default=120,
                        help="how long before close to take each market's quote")
    parser.add_argument("--window", type=int, default=20,
                        help="how far further back to look for that quote")
    parser.add_argument("--family", default=None, help="restrict to one series family")
    parser.add_argument("--by-family", action="store_true",
                        help="also report each family separately, since the bias "
                             "may live in some and not others")
    args = parser.parse_args()

    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    try:
        rows = await sample(
            conn, minutes_before=args.minutes_before,
            window=args.window, family=args.family,
        )
        if not rows:
            print("no settled market has a quote at that offset; "
                  "run scripts/backfill_settled.py first")
            return 1
        report(rows, label=f"{args.minutes_before}m before close")

        if args.by_family:
            families: dict[str, list[Any]] = defaultdict(list)
            for row in rows:
                families[row["fam"]].append(row)
            for fam, group in sorted(
                families.items(), key=lambda kv: -len(kv[1])
            )[:6]:
                if len(group) >= MIN_MARKETS:
                    report(group, label=fam)
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
