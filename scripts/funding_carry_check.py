"""Is perpetual funding a carry, or a directional bet with a coupon?

A carry trade collects a spread between two legs and is indifferent to the
price. A funding trade on Kalshi has only one leg, because the exchange lists no
spot and no dated future against which a perpetual could be hedged. Whatever
funding pays, the position carrying it is exposed to the whole price move.

That makes the question arithmetic rather than strategy. Divide the annualised
funding by the annualised volatility of the thing you must hold to collect it,
and the result is the best Sharpe such a trade could reach *before* any cost,
assuming the rate persists and is captured in full.

Measured across the board, the best market on the exchange returns 0.47 and
every other one falls between 0.01 and 0.14. The best is Bitcoin at +14.3% a
year against 31% volatility -- and its funding carries the same sign only 41% of
the time, so even that is not collectable: it is a mean over a series that keeps
changing direction and mostly sits at zero.

So this is not a carry. It is a directional position in a volatile asset that
pays a coupon two orders of magnitude smaller than the risk, and the platform's
one-times-leverage cap does not change the ratio -- leverage would scale both
terms together.

Run it before believing any funding-based strategy. The rates are published, the
volatility is measurable, and the ratio between them settles the question
without a backtest.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import time
from typing import Any

from kat.kalshi.perps.rest import PerpsREST
from kat.observability.logging import configure_logging

sys.path.insert(0, "scripts")

from perp_backtest import _client

#: Kalshi settles perpetual funding every four hours -- six times a day, not
#: twenty-four. Annualising an hourly rate overstates the carry fourfold.
PERIODS_PER_YEAR = 6 * 365

#: Hours of hourly candles used to estimate price volatility.
VOL_WINDOW_HOURS = 60

#: Below this many published rates a market has no history worth averaging.
MIN_RATES = 20


def _rate_of(entry: Any) -> float | None:
    raw = getattr(entry, "rate", None)
    if raw is None:
        raw = getattr(entry, "funding_rate", None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


async def _annualised_vol(rest: PerpsREST, ticker: str) -> float:
    """Annualised volatility from hourly closes, or zero when unknowable."""
    end_ts = int(time.time())
    try:
        candles = await rest.get_candlesticks(
            ticker, start_ts=end_ts - VOL_WINDOW_HOURS * 3600,
            end_ts=end_ts, period_interval=60,
        )
    except Exception:
        return 0.0
    closes: list[float] = []
    for candle in candles:
        close = (candle.get("price") or {}).get("close")
        if close is None:
            continue
        try:
            closes.append(float(close))
        except (TypeError, ValueError):
            continue
    if len(closes) < 6:
        return 0.0
    steps = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(steps) < 3:
        return 0.0
    return statistics.stdev(steps) * math.sqrt(24 * 365) * 100


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200,
                        help="published funding rates to average per market")
    args = parser.parse_args()

    configure_logging("WARNING")
    http = await _client()
    rest = PerpsREST(http)
    rows: list[tuple[float, str, int, float, float, float, float]] = []
    try:
        markets = [
            m for m in await rest.get_markets(limit=200, max_pages=3)
            if m.bid_cc and m.ask_cc
        ]
        for market in markets:
            try:
                history = await rest.get_historical_funding_rates(
                    market.ticker, limit=args.limit
                )
            except Exception:
                continue
            rates = [r for r in (_rate_of(h) for h in history) if r is not None]
            if len(rates) < MIN_RATES:
                continue

            mean = statistics.mean(rates)
            annualised = mean * PERIODS_PER_YEAR * 100
            # How often the rate keeps its sign. A mean over a series that keeps
            # flipping is not a rate anyone can collect, and zeros count against
            # it because a zero pays nothing.
            persistence = 100 * max(
                sum(1 for r in rates if r > 0),
                sum(1 for r in rates if r < 0),
            ) / len(rates)
            volatility = await _annualised_vol(rest, market.ticker)
            ratio = abs(annualised) / volatility if volatility > 0 else 0.0
            rows.append(
                (ratio, market.ticker, len(rates), mean,
                 annualised, persistence, volatility)
            )
    finally:
        await http.aclose()

    if not rows:
        print("no perpetual published enough funding history", file=sys.stderr)
        return 1

    rows.sort(reverse=True)
    print(f"\n  {'ticker':<14}{'n':>5}{'mean rate':>12}{'annualised':>12}"
          f"{'same sign':>11}{'ann. vol':>10}{'carry/vol':>11}")
    for ratio, ticker, n, mean, annualised, persistence, volatility in rows:
        print(f"  {ticker:<14}{n:>5}{mean:>+12.6f}{annualised:>+11.1f}%"
              f"{persistence:>10.0f}%{volatility:>9.0f}%{ratio:>11.2f}")

    best = rows[0]
    print(f"\n  Best on the board: {best[1]} at {best[0]:.2f}.")
    print("  carry/vol is the Sharpe an unhedged funding trade could reach")
    print("  before any cost, assuming the rate persists and is captured whole.")
    print("  Kalshi lists no spot, so the position cannot be hedged: the coupon")
    print("  is collected while carrying the entire price move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
