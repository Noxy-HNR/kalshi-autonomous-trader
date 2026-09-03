"""What a resting quote is worth on Polymarket, against what it was worth on Kalshi.

The equivalent survey on Kalshi produced the finding that ended the search there.
Market making needs two things at once -- a spread that covers costs, and a queue
short enough to reach the front of -- and Kalshi never offered both in the same
market. Where the volume was, the touch carried a median of 124,000 contracts and
a resting order was never reached except adversely. Where the queue was short,
nothing traded.

This asks the same two questions here, and adds the one that motivated the port:
what the round trip is actually worth once the maker is *paid* rather than
charged. Every row is priced twice, once on each venue's schedule, so the
comparison is arithmetic rather than recollection:

    Kalshi maker round trip      = spread - 2 * (0.07  * p * (1-p))
    Polymarket maker round trip  = spread + 2 * (0.0125 * p * (1-p))

Both legs resting earns the spread once, bid to ask.

What this cannot tell you
-------------------------
Whether it works. Adverse selection is what killed passive quoting on both
Kalshi instruments, and it is invisible in a snapshot: a resting order is reached
only when the market trades *through* it, which is a more selective and more
hostile subset of the flow than any survey sees. This measures whether the
economics are positive *before* that, which on Kalshi they were not, and which is
a necessary condition rather than a sufficient one.

The exchange publishes no book history and its candles carry no bid or ask, so
settling the sufficient condition means banking books as they happen.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from itertools import pairwise

from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT
from kat.kalshi.fees import estimate_fee_micros
from kat.observability.logging import configure_logging
from kat.polymarket.fees import polymarket_fee_micros
from kat.polymarket.rest import PolymarketREST

#: Price bands, in centicents. The fee is quadratic, so what a round trip is
#: worth depends on where in the range it sits.
BANDS = tuple(range(0, CC_PER_DOLLAR + 1, 1_000))


def _band(mid_cc: int) -> tuple[int, int] | None:
    for lo, hi in pairwise(BANDS):
        if lo <= mid_cc < hi:
            return lo, hi
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-markets", type=int, default=400)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    configure_logging("WARNING")
    rest = PolymarketREST()

    markets = await rest.get_open_markets(max_pages=4, categories=args.category)
    print(f"  {len(markets)} open markets on the venue")

    rows = []
    for market in markets[: args.max_markets]:
        try:
            book = await rest.get_book(market.slug)
        except Exception:
            continue
        if book is None:
            continue
        bid, ask = book.best_bid_cc, book.best_ask_cc
        if bid is None or ask is None or not 0 < bid < ask < CC_PER_DOLLAR:
            continue
        mid = (bid + ask) // 2
        spread = ask - bid
        one = QC_PER_CONTRACT

        # Signed, so a rebate arrives as negative and subtracts as income.
        poly_fee = polymarket_fee_micros(
            price_cc=mid, qty_qc=one, schedule=market.fees, is_taker=False
        ) // 100
        kalshi_fee = estimate_fee_micros(
            price_cc=mid, qty_qc=one, is_taker=False
        ).fee_micros // 100

        rows.append({
            "slug": market.slug,
            "mid": mid,
            "spread": spread,
            "queue": min(book.best_bid_size_qc, book.best_ask_size_qc),
            "poly_rt": spread - 2 * poly_fee,
            "kalshi_rt": spread - 2 * kalshi_fee,
            "tick": market.tick_size_cc,
        })

    if not rows:
        print("  no two-sided book found")
        return 1

    print(f"  {len(rows)} with a two-sided book\n")
    print(f"  {'mid band':>10}{'markets':>9}{'med spread':>12}"
          f"{'Polymarket RT':>15}{'Kalshi RT':>11}{'med queue':>11}")
    for lo, hi in pairwise(BANDS):
        band = [r for r in rows if _band(r["mid"]) == (lo, hi)]
        if len(band) < 3:
            continue
        print(f"  {f'{lo // 100}-{hi // 100}c':>10}{len(band):>9}"
              f"{statistics.median(r['spread'] for r in band):>12.0f}"
              f"{statistics.median(r['poly_rt'] for r in band):>+15.0f}"
              f"{statistics.median(r['kalshi_rt'] for r in band):>+11.0f}"
              f"{statistics.median(r['queue'] for r in band) / 100:>11,.0f}")

    poly_ok = [r for r in rows if r["poly_rt"] > 0]
    kalshi_ok = [r for r in rows if r["kalshi_rt"] > 0]
    reachable = [r for r in poly_ok if r["queue"] <= 50_000]
    print(f"\n  round trip positive on Polymarket : {len(poly_ok):>4} / {len(rows)}")
    print(f"  round trip positive on Kalshi's terms: {len(kalshi_ok):>4} / {len(rows)}")
    print(f"  ... and a queue under 500 contracts  : {len(reachable):>4}"
          "   <- both conditions at once, which Kalshi never offered")
    fine = sum(1 for r in rows if r["tick"] <= 10)
    print(f"  quoting at a tenth-cent tick         : {fine:>4} / {len(rows)}")
    if reachable:
        med = statistics.median(r["poly_rt"] for r in reachable)
        print(f"\n  median round trip where both hold: {med:+.0f}cc per contract")
        print("  Necessary, not sufficient: adverse selection is invisible here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
