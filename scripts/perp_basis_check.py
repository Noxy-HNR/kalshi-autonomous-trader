"""Does a perpetual's mid drift from its reference price by enough to trade?

PerpBasisStrategy has existed in this codebase since the beginning and had never
been run, because it requires an index price and none appeared to be available.
One is: Kalshi publishes ``reference_price`` on every perpetual alongside
``settlement_mark_price`` and ``liquidation_mark_price``. The basis is the gap
between what the book is quoting and what the exchange says the thing is worth.

The case for expecting one is decent. These are thin markets -- the busiest does
a few hundred million a day and most do under two million -- anchored to a
reference derived from venues that trade billions. A thin quote wandering from a
well-anchored reference is the textbook setup.

It does not wander far enough. The median gap across the board is 2.6 basis
points against a round trip of 8.4 to 28, and not one market on the exchange
shows a deviation exceeding its own cost -- the widest, at 14.4bp, sits inside an
18.1bp toll. Kalshi's perpetuals are already arbitraged to within a few basis
points, presumably by participants whose costs are lower than a small account's.

So the basis is real and an order of magnitude too small. This is a snapshot
rather than a time series, which is the honest limit of it: a distribution
measured over days could show a tail this cannot. But a median of 2.6bp with a
maximum of 14.4 does not suggest one worth waiting for, and the strategy should
not be armed on the hope of it.
"""
import asyncio
import sys

sys.path.insert(0, "scripts")
from perp_backtest import _client

from kat.kalshi.perps.rest import PerpsREST
from kat.observability.logging import configure_logging


async def snap(rest):
    out = {}
    for m in await rest.get_markets(limit=200, max_pages=3):
        raw = m.raw or {}
        ref = (raw.get("reference_price") or {}).get("price")
        mark = (raw.get("settlement_mark_price") or {}).get("price")
        if not m.bid_cc or not m.ask_cc or m.bid_cc >= m.ask_cc:
            continue
        try:
            ref_f = float(ref) if ref else None
            mark_f = float(mark) if mark else None
        except (TypeError, ValueError):
            continue
        mid = (m.bid_cc + m.ask_cc) / 2 / 10000.0
        spread_bp = (m.ask_cc - m.bid_cc) / ((m.bid_cc + m.ask_cc) / 2) * 1e4
        out[m.ticker] = (mid, ref_f, mark_f, spread_bp)
    return out

async def main():
    configure_logging("WARNING")
    http = await _client()
    rest = PerpsREST(http)
    try:
        s = await snap(rest)
        print(f"  {'ticker':<14}{'mid':>10}{'reference':>11}{'mark':>10}"
              f"{'mid-ref bp':>12}{'spread bp':>11}{'basis/spread':>13}")
        rows = []
        for tk, (mid, ref, mark, sp) in sorted(s.items()):
            if not ref or ref <= 0:
                print(f"  {tk:<14}{mid:>10.4f}{'n/a':>11}")
                continue
            dev = (mid - ref) / ref * 1e4
            rows.append((abs(dev), tk, mid, ref, mark, dev, sp))
        rows.sort(reverse=True)
        for adev, tk, mid, ref, mark, dev, sp in rows:
            ratio = adev / sp if sp > 0 else 0
            print(f"  {tk:<14}{mid:>10.4f}{ref:>11.4f}"
                  f"{(mark or 0):>10.4f}{dev:>+12.1f}{sp:>11.1f}{ratio:>13.2f}")
        if rows:
            devs = sorted(r[0] for r in rows)
            print(f"\n  median |mid - reference|: {devs[len(devs)//2]:.1f}bp")
            print("  a taker round trip costs spread + 8bp; a basis trade only")
            print("  pays if the deviation exceeds that AND then closes.")
    finally:
        await http.aclose()
asyncio.run(main())
