"""Replay the banked Polymarket books through the fill simulator.

Everything measured on this venue so far has been arithmetic on a snapshot: the
spread pays, the queue is short, the flow clears it 77% of the time it arrives.
None of that is a result. The Kalshi work produced exactly the same shape of
encouraging arithmetic and then lost money, because a resting order is reached
*when the market trades through it* -- which is a narrower and more hostile
subset of the flow than any average over intervals.

This is the measurement those numbers were leading to. It replays the banked
history through :mod:`kat.simulation.fills`, which assumes we sit behind every
contract already resting at our price and fills only when the market trades
past. That simulator has already refused five Kalshi strategies that looked good
on paper, which is the reason to trust it here.

Where the events come from
--------------------------
``quotes`` rows with ``source='polymarket'``: a book event from the bid, ask and
their sizes, and a print from the *difference* in the cumulative traded counter
since that market's previous row. The venue publishes no trades endpoint, so
differencing the counter is the only flow available.

The print is placed at ``last_cc``, the venue's last traded price, rather than
at the mid. Which price a print lands on decides whether a resting order is
reached at all -- the first Kalshi candle feed placed them at the mid and
reported twenty-one fills where there should have been a hundred and nineteen.

What this still cannot see
--------------------------
Sub-interval sequence. A pass takes about thirty seconds, so a book that moved
and came back inside one interval reads as a book that never moved, and a print
is attributed to a single price rather than to the path it actually took. That
blurs adverse selection in the direction of *understating* it: the sharpest
moves against a resting quote are the fastest ones.

So a negative result here is conclusive and a positive one is provisional.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import pathlib
import re
from typing import Any

from kat.common.enums import MarketType
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.config.settings import AppSettings, EventRiskSettings
from kat.observability.logging import configure_logging
from kat.polymarket.fees import PolymarketFeeSchedule
from kat.simulation.costs import CostModel
from kat.simulation.engine import (
    BacktestConfig,
    BacktestEngine,
    EventFeed,
    book_event,
    trade_event,
)
from kat.strategies.event.passive_quote import PassiveQuoteStrategy

#: A gap longer than this is a hole in collection, not a quiet market. The
#: counter is cumulative, so differencing across a collector restart would
#: attribute hours of trading to one interval and invent a burst.
MAX_GAP_S = 300

#: The collector polls about every thirty seconds, so the book a strategy sees
#: is up to that old. The live freshness bar of five seconds would refuse every
#: decision and measure a strategy standing still.
MAX_DATA_AGE_MS = 90_000

#: Every market on this venue publishes the same coefficient, but it is read
#: from the market rather than assumed everywhere else, so the run states which
#: one it used rather than burying it.
FEES = PolymarketFeeSchedule.from_market({"feeCoefficient": "0.06"})


class ReplayQuoteStrategy(PassiveQuoteStrategy):
    """The quoting strategy, told how coarse this history is.

    Live, five seconds of silence on a websocket means something is wrong. Here
    the book is a poll thirty seconds old by construction, which is the data's
    resolution rather than a fault -- the same override the Kalshi candle feed
    needed, and the same reason results from it are an upper bound.
    """

    MAX_DATA_AGE_MS = MAX_DATA_AGE_MS


async def _load(hours: int) -> list[Any]:
    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetch(
            """
            select q.ticker, q.ts, q.bid_cc, q.ask_cc,
                   q.bid_size_qc, q.ask_size_qc,
                   q.last_cc, q.volume_qc,
                   -- Full depth where it was banked. Replaying a one-level book
                   -- makes total_depth_qc equal to the touch, and the risk
                   -- engine's min_book_depth_qc then reads a market carrying
                   -- 494,100 centi-contracts across its levels as one carrying
                   -- 176 -- which refused 72,467 of 101,379 intents and read as
                   -- "the venue is too thin" rather than "the history is."
                   ob.bids as ob_bids, ob.asks as ob_asks,
                   lag(q.volume_qc) over w as prev_volume,
                   lag(q.ts)        over w as prev_ts
            from quotes q
            left join orderbook_snapshots ob
              on ob.source = 'polymarket'
             and ob.ticker = q.ticker
             and ob.ts = q.ts
            where q.source = 'polymarket'
              and q.bid_cc is not null and q.ask_cc is not null
              and q.ts > now() - make_interval(hours => $1)
              -- Only markets the collector *followed*. A market that was merely
              -- swept has a book and no traded counter, so no print can ever be
              -- reconstructed for it and a quote there cannot fill by
              -- construction. Including them does not make the test harder, it
              -- makes it meaningless: three thousand un-fillable markets absorb
              -- the inventory and position limits that the sixty measurable
              -- ones need, and the run reports zero fills for a reason that has
              -- nothing to do with the strategy.
              and q.ticker in (
                  select ticker from quotes
                  where source = 'polymarket' and volume_qc is not null
                    and ts > now() - make_interval(hours => $1)
                  group by ticker having count(*) > 5
              )
            window w as (partition by q.ticker order by q.ts)
            order by q.ts
            """,
            hours,
        )
    finally:
        await conn.close()


def _levels(raw: Any) -> list[tuple[int, int]]:
    """Banked book levels as (price_cc, size_qc), or empty when absent."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [(int(p), int(q)) for p, q in raw if q]


def _build_feed(rows: list[Any]) -> tuple[EventFeed, int, int]:
    events: list[Any] = []
    seq = prints = deep = 0
    for row in rows:
        bid, ask = row["bid_cc"], row["ask_cc"]
        if not 0 < bid < ask < CC_PER_DOLLAR:
            continue
        ts_ms = int(row["ts"].timestamp() * 1000)
        # Prefer the banked levels; fall back to the touch for rows collected
        # before full books were being written, which is most of the history.
        bid_levels = _levels(row["ob_bids"]) or [(bid, row["bid_size_qc"] or 0)]
        ask_levels = _levels(row["ob_asks"]) or [(ask, row["ask_size_qc"] or 0)]
        if row["ob_bids"]:
            deep += 1
        seq += 1
        events.append(
            book_event(
                row["ticker"], info_ts_ms=ts_ms,
                bids=bid_levels, asks=ask_levels, seq=seq,
            )
        )
        prev_v, prev_ts = row["prev_volume"], row["prev_ts"]
        if prev_v is None or prev_ts is None or row["volume_qc"] is None:
            continue
        gap_s = (row["ts"] - prev_ts).total_seconds()
        delta = row["volume_qc"] - prev_v
        price = row["last_cc"]
        if delta <= 0 or not 0 < gap_s <= MAX_GAP_S or not price:
            continue
        if not 0 < price < CC_PER_DOLLAR:
            continue
        seq += 1
        prints += 1
        events.append(
            trade_event(
                row["ticker"], info_ts_ms=ts_ms + 1,
                price_cc=price, qty_qc=delta, seq=seq,
            )
        )
    events.sort(key=lambda e: (e.info_ts_ms, e.seq))
    return EventFeed(events, name="polymarket"), prints, deep


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--capital", type=float, default=200.0)
    parser.add_argument("--report", default=None,
                        help="append the summary to this file with a timestamp, "
                             "so a scheduled run leaves a record rather than "
                             "writing to a console nobody is watching")
    parser.add_argument("--min-depth-qc", type=int, default=None)
    parser.add_argument("--max-queue-qc", type=int, default=None)
    parser.add_argument("--quote-qc", type=int, default=None)
    parser.add_argument("--charge-maker-as-taker", action="store_true",
                        help="price every fill at the taker rate, discarding the "
                             "rebate; the pessimistic bound")
    args = parser.parse_args()

    configure_logging("WARNING")

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    rows = await _load(args.hours)
    if not rows:
        print("no banked Polymarket books in that window")
        return 1

    overrides = {
        k: v for k, v in (
            ("min_depth_qc", args.min_depth_qc),
            ("max_queue_qc", args.max_queue_qc),
            ("quote_qc", args.quote_qc),
        ) if v is not None
    }
    feed, prints, deep = _build_feed(rows)
    markets = len({r["ticker"] for r in rows})
    span_h = (rows[-1]["ts"] - rows[0]["ts"]).total_seconds() / 3600
    say(f"  feed     {len(feed):,} events over {markets:,} markets, "
        f"{span_h:.1f}h, {prints:,} prints")
    say(f"  depth    {deep:,} books with full levels, "
        f"{len(rows) - deep:,} touch-only")
    if overrides:
        say(f"  params   {overrides}")
    say(f"  fees     taker {FEES.taker_theta}, maker {FEES.maker_theta}"
          f"{'  (rebate discarded)' if args.charge_maker_as_taker else ''}")

    engine = BacktestEngine(
        feed=feed,
        strategies=[ReplayQuoteStrategy(overrides)],
        config=BacktestConfig(
            market_type=MarketType.EVENT,
            initial_capital_micros=int(args.capital * 1_000_000),
            costs=CostModel(
                polymarket_fees=FEES,
                charge_maker_as_taker=args.charge_maker_as_taker,
                # The live loop cancels a working order after a minute. A
                # backtest that never cancels is not measuring the same
                # strategy, and lets unfilled quotes accumulate exposure.
                order_ttl_ms=60_000,
            ),
            decision_interval_ms=30_000,
        ),
        settings=AppSettings(),
        event_risk_settings=EventRiskSettings(),
    )
    result = engine.run()

    trades = getattr(result, "trades", None) or []
    distinct = {
        (t.ticker, t.entry_ms, t.exit_ms, t.qty_qc, t.net_pnl_micros) for t in trades
    }
    m = result.metrics
    say()
    # The gap between "a print could have filled us" and "a print did" is the
    # whole question, so the order lifecycle is reported rather than inferred.
    report = engine.fills.report.to_dict()
    interesting = {
        k: v for k, v in report.items()
        if k in ("expired_orders", "unfilled_orders", "rejected_post_only",
                 "fill_count", "maker_fills", "taker_fills")
    }
    say(f"  orders   {interesting}")
    # Where the intents went. A strategy producing a hundred thousand intents
    # and two thousand orders is being refused by risk, not by the market, and
    # which check refuses them is the difference between a finding and a
    # misconfiguration.
    decisions = sorted(engine.risk_decisions.items(), key=lambda kv: -kv[1])[:5]
    say(f"  risk     {', '.join(f'{k}={v:,}' for k, v in decisions)}")
    say(f"  fills    {len(engine.fills.fills):,}")
    say(f"  trades   {len(trades):,} booked, {len(distinct):,} distinct")
    say(f"  return   {m.total_return:+.4%}")
    say(f"  win rate {m.win_rate:.0%}")
    say(f"  fees     ${m.total_fees_micros / 1e6:+.4f}   (negative is income)")
    say(f"  maker    {m.maker_fill_pct:.0%} of fills")
    for strategy in engine.strategies:
        health = strategy.health()
        skips = sorted(health.get("skips", {}).items(), key=lambda kv: -kv[1])[:4]
        say(f"  {strategy.strategy_id}: intents={health.get('intents', 0):,} "
              f"{', '.join(f'{k}={v}' for k, v in skips)}")

    if args.report:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        path = pathlib.Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== {stamp} =====\n")
            fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
