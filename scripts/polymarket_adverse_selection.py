"""What a resting quote is worth on Polymarket US, cut by spread.

Adverse selection is the question every passive structure on this platform has
died of, and it cannot be seen in a snapshot: a resting order is reached only
when the market trades *through* it, which is a narrower and more hostile subset
of the flow than any survey sees. This measures it directly.

The method deliberately drops the strategy and the risk engine. For every print
that would have reached a quote resting at the touch, it books the fill that
implies and follows the mid afterwards. That gives a few hundred fills to cut by
spread, where routing the same question through the live machinery gives twenty.

Two properties make the number honest:

* The mark-out is measured **from our own fill price**, so the half-spread we
  captured is already inside it. A buy filled at the bid against a mid one tick
  higher starts half a spread ahead, and what happens next is the adverse
  selection. Positive means the quote was worth placing.
* A print *at* our price has to exhaust the queue ahead before it reaches us; a
  print *through* it fills us whatever is queued. That is the fill simulator's
  rule, reproduced here rather than assumed away.

What it cannot see is sub-interval sequence. The collector polls about every
thirty seconds, so a book that moved and came back reads as one that never
moved, and the fastest adverse moves -- the ones that hurt most -- are invisible.
That bias runs one way: these numbers **understate** adverse selection, so a
negative result here is conclusive and a positive one is provisional.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import collections
import datetime as dt
import re
import statistics
from typing import Any

from kat.observability.logging import configure_logging
from kat.polymarket.fees import PolymarketFeeSchedule, polymarket_fee_micros

#: A gap longer than this is a hole in collection, not a quiet market. The
#: counter is cumulative, so differencing across a restart would attribute hours
#: of trading to one interval and invent a burst.
MAX_GAP_S = 300

#: Minutes after the fill at which the mid is read.
HORIZONS = (1, 5, 30)

#: Spread bands, in centicents. The tick here is 10cc.
BUCKETS = ((0, 100), (100, 200), (200, 400), (400, 1_000), (1_000, 10_000))

SQL = """
select ticker, ts, bid_cc, ask_cc, bid_size_qc, ask_size_qc, last_cc, volume_qc
from quotes
where source = 'polymarket'
  and bid_cc is not null and ask_cc is not null
  and ts > $1 and ts <= $2
order by ts
"""


async def _load(start: dt.datetime, end: dt.datetime) -> list[dict[str, Any]]:
    import asyncpg

    from kat.config.settings import AppSettings

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    try:
        return [dict(r) for r in await conn.fetch(SQL, start, end)]
    finally:
        await conn.close()


def _mids(rows: list[dict[str, Any]]) -> dict[str, tuple[list[int], list[int]]]:
    series: dict[str, tuple[list[int], list[int]]] = {}
    for row in rows:
        times, mids = series.setdefault(row["ticker"], ([], []))
        times.append(int(row["ts"].timestamp() * 1000))
        mids.append((row["bid_cc"] + row["ask_cc"]) // 2)
    return series


def _mid_at(series, ticker: str, ts_ms: int, minutes: int) -> int | None:
    entry = series.get(ticker)
    if entry is None:
        return None
    times, mids = entry
    index = bisect.bisect_left(times, ts_ms + minutes * 60_000)
    # None rather than the last known mid: the history ending before the horizon
    # is missing data, and carrying the last price forward would report it as a
    # flat market.
    return None if index >= len(times) else mids[index]


def _fills(rows: list[dict[str, Any]], schedule: PolymarketFeeSchedule) -> list[dict[str, Any]]:
    """Every print a continuous touch quote would have been reached by."""
    previous: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        before = previous.get(row["ticker"])
        previous[row["ticker"]] = row
        if before is None or row["volume_qc"] is None or before["volume_qc"] is None:
            continue
        gap = (row["ts"] - before["ts"]).total_seconds()
        traded = row["volume_qc"] - before["volume_qc"]
        price = row["last_cc"]
        if traded <= 0 or not 0 < gap <= MAX_GAP_S or not price:
            continue
        bid, ask = before["bid_cc"], before["ask_cc"]
        if not bid or not ask or not 0 < bid < ask:
            continue

        # Which side of our own quote the print could have reached.
        if price <= bid:
            is_buy, ours, queue = True, bid, before["bid_size_qc"] or 0
        elif price >= ask:
            is_buy, ours, queue = False, ask, before["ask_size_qc"] or 0
        else:
            continue  # inside the spread: reaches neither side at the touch
        if price == ours and traded <= queue:
            continue  # the queue ahead absorbed the whole print

        out.append({
            "ticker": row["ticker"],
            "ts_ms": int(row["ts"].timestamp() * 1000),
            "is_buy": is_buy,
            "price_cc": ours,
            "spread_cc": ask - bid,
            "through": price != ours,
            # Signed, so income is negative: subtracting it adds the rebate.
            "rebate_cc": -polymarket_fee_micros(
                price_cc=ours, qty_qc=100, schedule=schedule, is_taker=False
            ) / 100.0,
        })
    return out


def _markout(series, fill: dict[str, Any], minutes: int) -> float | None:
    mid = _mid_at(series, fill["ticker"], fill["ts_ms"], minutes)
    if mid is None:
        return None
    move = (mid - fill["price_cc"]) if fill["is_buy"] else (fill["price_cc"] - mid)
    return move + fill["rebate_cc"]


def _summarise(sample: list[float]) -> str:
    if len(sample) < 2:
        return f"n={len(sample)}".rjust(19)
    mean = statistics.mean(sample)
    ci = 1.96 * statistics.stdev(sample) / (len(sample) ** 0.5)
    return f"{mean:+7.0f} +-{ci:5.0f}cc".rjust(19)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--end", default=None,
                        help="ISO instant to measure back from; defaults to now. "
                             "Pin it to make a number reproducible tomorrow")
    args = parser.parse_args()
    configure_logging("WARNING")

    end = dt.datetime.fromisoformat(args.end) if args.end else dt.datetime.now(dt.UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.UTC)
    start = end - dt.timedelta(hours=args.hours)

    rows = await _load(start, end)
    if not rows:
        print("no banked Polymarket books in that window")
        return 1

    schedule = PolymarketFeeSchedule.from_market({"feeCoefficient": "0.06"})
    fills = _fills(rows, schedule)
    series = _mids(rows)
    span_h = (rows[-1]["ts"] - rows[0]["ts"]).total_seconds() / 3600
    markets = len({r["ticker"] for r in rows})

    print(f"  window   {rows[0]['ts']:%Y-%m-%d %H:%M} -> {rows[-1]['ts']:%Y-%m-%d %H:%M}Z "
          f"({span_h:.1f}h, {len(rows):,} books, {markets:,} markets)")
    if not fills:
        print("  no print in this window would have reached a quote at the touch")
        return 0
    through = sum(1 for f in fills if f["through"])
    print(f"  ceiling  {len(fills):,} fills a continuous touch quote would have taken "
          f"({len(fills) / span_h * 24:,.0f}/day; {through:,} traded through us)")
    print()
    print(f"  {'spread at fill':<16} {'n':>5}  "
          + "  ".join(f"{m}m mark-out".rjust(19) for m in HORIZONS))
    for low, high in BUCKETS:
        bucket = [f for f in fills if low <= f["spread_cc"] < high]
        if not bucket:
            continue
        cells = [
            _summarise([m for m in (_markout(series, f, h) for f in bucket) if m is not None])
            for h in HORIZONS
        ]
        print(f"  {f'{low}-{high}cc':<16} {len(bucket):>5}  " + "  ".join(cells))

    overall = [m for m in (_markout(series, f, 5) for f in fills) if m is not None]
    if len(overall) > 1:
        positive = sum(1 for m in overall if m > 0)
        print()
        print(f"  all fills, 5m: {_summarise(overall).strip()} per contract, "
              f"{positive}/{len(overall)} positive ({positive / len(overall):.0%})")
        markets_hit = collections.Counter(f["ticker"] for f in fills)
        print(f"  across {len(markets_hit):,} markets; busiest contributed "
              f"{markets_hit.most_common(1)[0][1]} of {len(fills):,} fills")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
