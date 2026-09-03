# What was measured, and what it cost to trade

This records eleven structures tested on Kalshi across both instruments, so that
none of them is searched twice. Every figure here came from this repository's own
tooling and can be reproduced by the script named beside it.

Nothing is armed. `KAT_ENVIRONMENT=paper`, `KAT_LIVE_TRADING_ARMED=false`.

## The number that governs everything

A round trip has to be paid before an edge is an edge.

| | event contract | perpetual |
| --- | --- | --- |
| fee | `0.07·C·P·(1−P)` — 175cc at 50c, **350bp** | **2bp** maker, **4bp** taker |
| minimum tick | 100cc — **3,333bp** on a 3c contract | 1cc — **0.8bp** at $1.26 |
| typical round trip | 136–550cc | 8.4–28bp |
| touch queue | median **124,000 contracts** | $49–$100 of notional |

Event contracts are two orders of magnitude more expensive than perpetuals. Every
negative result on events is a statement about that column and says nothing about
the other.

## Event contracts

| structure | result | why | script |
| --- | --- | --- | --- |
| Directional taker | 0 wins in 90 round trips | A 30-minute move is smaller than the toll to capture it, even with a 0.949 walk-forward AUC model | `scripts/backtest.py` |
| Complementary-pair arbitrage | Does not exist | The NO book is the YES book reflected; buying NO costs `$1 − yes_bid` by construction | — |
| Cross-leg (exclusive-set) arbitrage | 26 chances in 464,606 aligned minutes | Exhaustiveness verified from outcomes, not the `mutually_exclusive` flag: exactly one leg settles YES in 220/220 MLB, 172/172 WNBA, 47/47 NFL, 90/90 UCL | `scripts/candle_feed.py` |
| Market making | −0.49%, 74% win rate | Winning three-quarters of round trips and losing money is adverse selection. Where the volume is, the queue is 124,000 deep; where the queue is short, 0.3% of the volume trades | `scripts/candle_feed.py` |
| Settlement model | 0.71 AUC vs the market's **0.7764** | The model was worse than the quote it would have traded against. 0.5 is not the benchmark | `scripts/price_baseline.py` |
| Favourite–longshot bias | Flips sign as the sample grows | +114bp on 2,050 markets became −826bp on more. Pooling per-minute rows had inflated `n` by 1,000× | `scripts/settlement_bias.py` |
| Single-leg mispricing | −8 to −830cc per contract | The complementary leg genuinely predicts the other — monotonically — but only 1,121 of 750,074 observations show an edge past cost | — |

The unifying finding: **the market prices settlements well** (0.7764 AUC, rising
to 0.8417 in the final hour) and **charges more than the mispricing is worth**.

## Perpetuals

Costs are ~100× lower. Every structure still fails, for different reasons.

| structure | result | why | script |
| --- | --- | --- | --- |
| Passive quoting | −7.49% to −17.88%, 39% win rate | Adverse selection survives cheap fees. Fees were $4.42 of a $35.60 loss — waiving them entirely would not have saved it | `scripts/perp_backtest.py` |
| Directional taker | 55.6% accuracy against a **126.3%** bar | The break-even bar is `p = 0.5 + cost/(2·move)`. Selecting volatile windows raises the move *and* the spread, and they cancel | `scripts/perp_direction.py` |
| Model + passive entry | −2.65% over 800 trades | The decisive experiment. Model scored ~65% on windows it fired on; **50.0%** on the trades that actually filled. A resting entry selects against the trades the forecast got right | `scripts/perp_model_backtest.py` |
| Funding carry | Best Sharpe **0.47** before costs | Kalshi lists no spot, so the position cannot be hedged: the coupon is collected while carrying the whole price move. BTC pays +14.3%/yr against 31% vol, and holds sign 41% of the time | `scripts/funding_carry_check.py` |
| Basis vs reference price | Median **2.6bp** against an 8.4–28bp round trip | Not one market deviates further than its own cost. Already arbitraged by participants with lower costs | `scripts/perp_basis_check.py` |

Two dimensions were swept rather than assumed:

- **Horizon** (`scripts/perp_horizon_sweep.py`). The bar falls from 95% to 55.4%
  between 15 minutes and 12 hours, exactly as `cost/√time` predicts — but AUC
  decays from 0.5130 to 0.4940 on the way. The signal dies faster than the toll.
- **Cross-market features.** BTC leading altcoins is robust elsewhere; here,
  board drift and BTC/ETH leader returns took out-of-sample AUC from 0.5351 to
  **0.5031**. Not kept.

## The two doors, and why both are shut

The clearest single result in the project:

- **Cross the entry** — keep 65% accuracy, need 72.6% (later 126.3% on a longer
  sample). Short.
- **Rest the entry** — need only 63.5%, but accuracy collapses to **50.0%**,
  because a resting bid fills when the price comes *back* to it (the forecast was
  wrong) and is missed when the price runs (it was right).

The cheap entry and the accuracy cannot be had together. That is a property of
resting on a directional forecast, not of this model or venue.

## The information door, and why it is shut too

Every structure above searches for an edge *inside the price data*. The obvious
remaining category is an edge from outside it: Kalshi markets settle on
real-world measurements, and for weather the US government publishes both the
forecast and the observations for free, at the exact stations Kalshi settles on.
``api.weather.gov`` needs no key.

Two versions of the idea, and the market has already taken both.

**Forecasting.** NWS publishes a next-day high for each station. But so does
everyone's phone, and the market reads it: 12-24 hours out the price scores
0.80 AUC, which is roughly what next-day forecast skill supports.

**Nowcasting.** These settle on a station's observed daily *high*, and by
mid-afternoon that number is largely already set. A market slow to converge on
an outcome that is no longer uncertain would be free money. It is not slow:

| family      | 12-24h out | 6-12h  | 3-6h   |
| ----------- | ---------- | ------ | ------ |
| KXHIGHNY    | 0.7995     | 0.9021 | 0.9950 |
| KXHIGHLAX   | 0.8536     | 0.9847 | --     |
| KXHIGHMIA   | 0.8513     | 0.9224 | --     |
| KXHIGHCHI   | 0.8000     | 0.8875 | --     |

They close at 05:00-06:00 UTC, so "3-6h out" is evening local -- hours after the
day's high is set. New York prices it at 0.9950. Somebody is watching the
observations, and the price already contains them.

The check cost two API calls and one existing script, which is the argument for
having built ``price_baseline.py``: an information edge is only an edge where the
price does not already hold the information, and that is measurable before any
data pipeline is written.

## Six results that were wrong

Every one looked real. Each is recorded because the detection method is worth
more than the finding would have been.

| claimed | actual | cause |
| --- | --- | --- |
| +484% | +4.84% | `qty_qc × price_cc` is already micros; dividing by 100 again |
| 144 trades, 0 wins | 5 distinct trades | One position closed by many fills sharing `entry_ms` |
| +190bp longshot bias | ±596cc, straddles zero | 2,050 markets counted 2.17M times |
| +2.15% perp quoting | −7.49% | Orders never expired; exposure reached $3.2 quadrillion and the leverage invariant blocked everything after |
| "misses by 7 points" | bar is 126.3% | A four-day test window |
| Market making measurable | It was not | Prints placed at the mid reach no resting order on either side |

**The tell is always the same**: a confidence interval far too tight for the
sample, or a sample counted more times than it happened. Three helpers now
enforce the discipline — `scripts/price_baseline.py` (what the market scores, so
a model is never read against 0.5), `scripts/settlement_bias.py` (one row per
market, never per minute), and `scripts/perp_horizon_sweep.py` (windows stepped
by the horizon, never overlapped).

## What would change the answer

1. **Real perpetual book depth.** Every perp result rests on candle summaries —
   one-minute bid/ask closes with depth sampled once and held constant. The
   engine gates the perp domain on credentials that have never been set:

   ```
   KAT_KALSHI__PERP_CREDENTIALS__API_KEY_ID=...
   KAT_KALSHI__PERP_CREDENTIALS__PRIVATE_KEY_PATH=...   # outside the repo
   ```

   With those, the engine banks real books and microprice quoting — the standard
   mitigation for the adverse selection measured above — becomes testable.

2. **A cheaper venue.** The fill simulator, fee models, walk-forward splits and
   artifact detection are venue-agnostic. The costs are not.

## Reproducing any of it

```bash
.venv/Scripts/python.exe scripts/price_baseline.py --by-family
.venv/Scripts/python.exe scripts/funding_carry_check.py
.venv/Scripts/python.exe scripts/perp_basis_check.py
.venv/Scripts/python.exe scripts/perp_backtest.py --hours 336 --cache train.pkl
.venv/Scripts/python.exe scripts/perp_direction.py train.pkl test.pkl
```

`scripts/backfill_settled.py` banks settled-market history from Kalshi's
candlesticks — 1,014 labelled markets across 13 close dates, from history that
already existed rather than data waited for.
