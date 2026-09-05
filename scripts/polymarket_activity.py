"""Does this venue trade? Measured over a broad sample, not a chosen one.

The collector follows sixty markets ranked by edge per contract of queue ahead,
which is the right ranking for *trading* and the wrong one for *measuring*. It
favours short queues, a short queue usually means few people are quoting, and few
quoters correlates with few traders. Asking that set whether the venue trades
would be asking the markets most likely to say no.

So this takes two readings of the cumulative traded counter across a wide slice
of the open board, separated by a wait, and counts what moved. No ranking, no
edge filter, no selection at all beyond being open and quoted.

The counter is the only flow the venue exposes: there is no public trades
endpoint, and the candlestick endpoint that carries volume is authenticated and
publishes no bid or ask anyway.

What a "no" here would mean
---------------------------
Not that the port was wrong -- the fee arithmetic is real and a maker is paid
here where Kalshi charges. It would mean the same thing the Kalshi survey found
one layer down: positive economics on markets nobody trades. Being paid to quote
is worth nothing if nothing ever lifts the quote, and that is a property of the
venue rather than of the strategy, so it is worth knowing before a fortnight of
collection rather than after.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from kat.observability.logging import configure_logging
from kat.polymarket.rest import PolymarketREST


async def _snapshot(
    rest: PolymarketREST, slugs: list[str], *, rate_per_s: float
) -> dict[str, int]:
    """Cumulative traded count per market, skipping what will not answer."""
    out: dict[str, int] = {}
    gap = 1.0 / rate_per_s
    loop = asyncio.get_running_loop()
    next_at = loop.time()
    for slug in slugs:
        delay = next_at - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        next_at = loop.time() + gap
        try:
            shares, _ = await rest.get_flow(slug)
        except Exception:
            continue
        if shares is not None:
            out[slug] = shares
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=300)
    parser.add_argument("--wait-s", type=int, default=900,
                        help="seconds between the two readings")
    parser.add_argument("--rate-per-s", type=float, default=5.0)
    args = parser.parse_args()

    configure_logging("WARNING")
    rest = PolymarketREST()

    markets = await rest.get_open_markets(max_pages=4)
    tradeable = [m for m in markets if m.is_tradeable][: args.markets]
    slugs = [m.slug for m in tradeable]
    print(f"  sampling {len(slugs)} open markets, two readings {args.wait_s}s apart")

    started = time.time()
    first = await _snapshot(rest, slugs, rate_per_s=args.rate_per_s)
    elapsed = time.time() - started
    remaining = args.wait_s - elapsed
    print(f"  first reading: {len(first)} markets report a counter "
          f"({elapsed:.0f}s); waiting {max(remaining, 0):.0f}s")
    if remaining > 0:
        await asyncio.sleep(remaining)

    second = await _snapshot(rest, slugs, rate_per_s=args.rate_per_s)
    span_s = time.time() - started

    both = sorted(set(first) & set(second))
    moved = [(s, second[s] - first[s]) for s in both if second[s] > first[s]]
    total_qc = sum(d for _, d in moved)

    print(f"\n  {len(both)} markets read twice over {span_s / 60:.1f} minutes")
    print(f"  markets that traded : {len(moved)}  "
          f"({100 * len(moved) / max(len(both), 1):.1f}%)")
    print(f"  contracts traded    : {total_qc / 100:,.0f}")
    if moved:
        moved.sort(key=lambda kv: -kv[1])
        print(f"\n  {'market':<46}{'contracts':>11}")
        for slug, delta in moved[:10]:
            print(f"  {slug[:46]:<46}{delta / 100:>11,.0f}")
        per_hour = total_qc / 100 / max(span_s / 3600, 1e-9)
        print(f"\n  venue-wide across this sample: {per_hour:,.0f} contracts/hour")
        print(f"  per trading market: {total_qc / 100 / len(moved):,.0f} contracts "
              f"in {span_s / 60:.0f} minutes")
    else:
        print("\n  Nothing traded anywhere in the sample. A maker rebate on a")
        print("  venue with no takers pays exactly nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
