"""Does anything actually trade here, and would it ever reach our quote?

The survey said the maker economics on this venue are positive before adverse
selection: all 355 live two-sided books pay a passive round trip, against 169 of
the same books under Kalshi's schedule. That is a statement about prices. It says
nothing about whether anyone trades at them, and two measurements taken while
setting up the collector disagreed sharply about that:

* 78% of open markets report zero ``sharesTraded``, and the exchange's own
  busiest show 0-1,828 against open interest of 71,000 to 427,000.
* The banked books move constantly -- touch size changes in 74% of observations,
  and 2,016 of 3,560 markets have moved their touch at least once.

Both can be true at once, and the combination is the worst case for a market
maker: makers quoting at each other with almost no takers. Queue competition
without flow is what killed passive quoting on Kalshi's thin-queue markets, and
a rebate does not repair it -- being paid to quote is worth nothing if nothing
ever lifts the quote.

This settles it with per-interval flow rather than either proxy. The venue
publishes no trades endpoint, so the collector banks the cumulative
``sharesTraded`` beside each book and this differences it.

The number that decides it
--------------------------
Not volume. **Whether flow exceeds the queue ahead of us.** Fills go in time
order: a resting order at the touch is reached only after everything already
resting at that price has traded. So a market that trades a hundred contracts an
hour behind a queue of ten thousand never fills us, and a market that trades ten
behind a queue of five fills us twice.

That ratio, per interval, is what this reports. It is the same question the
Kalshi survey asked and answered badly, by measuring volume and queue separately
and never dividing one by the other.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from typing import Any

from kat.config.settings import AppSettings

#: Intervals longer than this are treated as a gap in collection rather than a
#: quiet market: the differenced counter would attribute a restart's worth of
#: trading to one interval and invent a burst that never happened.
MAX_INTERVAL_S = 300


async def _rows(conn: Any, hours: int) -> list[Any]:
    return await conn.fetch(
        """
        select ticker, ts, bid_cc, ask_cc, bid_size_qc, ask_size_qc,
               volume_qc, last_cc,
               lag(volume_qc) over w as prev_volume,
               lag(ts)         over w as prev_ts,
               lag(bid_size_qc) over w as prev_bid_size,
               lag(ask_size_qc) over w as prev_ask_size
        from quotes
        where source = 'polymarket'
          and volume_qc is not null
          and ts > now() - make_interval(hours => $1)
        window w as (partition by ticker order by ts)
        order by ticker, ts
        """,
        hours,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    try:
        rows = await _rows(conn, args.hours)
    finally:
        await conn.close()

    if not rows:
        print("no flow banked yet; the collector fetches it only while following,"
              " which starts after the opening sweep")
        return 1

    intervals = traded = 0
    volumes: list[int] = []
    reach: list[float] = []
    per_market: dict[str, int] = {}
    for r in rows:
        if r["prev_volume"] is None or r["prev_ts"] is None:
            continue
        gap_s = (r["ts"] - r["prev_ts"]).total_seconds()
        if not 0 < gap_s <= MAX_INTERVAL_S:
            continue
        intervals += 1
        # The counter is cumulative and monotone; a decrease means the market
        # reset it or we crossed a restart, and inventing a negative trade would
        # be worse than dropping the interval.
        delta = (r["volume_qc"] or 0) - (r["prev_volume"] or 0)
        if delta <= 0:
            continue
        traded += 1
        volumes.append(delta)
        per_market[r["ticker"]] = per_market.get(r["ticker"], 0) + delta
        queue = min(r["prev_bid_size"] or 0, r["prev_ask_size"] or 0)
        if queue > 0:
            reach.append(delta / queue)

    print(f"\n  {intervals:,} intervals over {len({r['ticker'] for r in rows})} markets, "
          f"last {args.hours}h")
    print(f"  intervals with any trade : {traded:,}  "
          f"({100 * traded / max(intervals, 1):.2f}%)")
    if not volumes:
        print("\n  Nothing traded in any interval. A resting quote on this venue")
        print("  is not competing for queue position, it is waiting alone.")
        return 0

    volumes.sort()
    print(f"  contracts per trading interval: median "
          f"{volumes[len(volumes) // 2] / 100:,.0f}, "
          f"max {volumes[-1] / 100:,.0f}")
    print(f"  markets that traded at all    : {len(per_market)}")

    if reach:
        reach.sort()
        cleared = sum(1 for x in reach if x >= 1.0)
        print("\n  flow / queue-ahead, per trading interval:")
        for pct in (50, 75, 90, 99):
            print(f"     p{pct:<3}{reach[min(len(reach) - 1, len(reach) * pct // 100)]:>10.3f}")
        print(f"     intervals where flow cleared the queue: {cleared:,} / {len(reach):,}"
              f"  ({100 * cleared / len(reach):.2f}%)")
        print("\n  A resting order is reached only when the flow ahead of it trades.")
        print("  That last line is the fill rate a passive quote can actually expect;")
        print("  everything above it is volume that went to somebody earlier in the queue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
