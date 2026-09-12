# What was measured, and what it cost to trade

This records eleven structures tested on Kalshi across both instruments, and the
first measurements on Polymarket US, so that none of them is searched twice. Every figure here came from this repository's own
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

## The venue changed

Everything above is a finding about Kalshi, not about prediction markets. Twelve
structures failed there for one recurring reason -- the tolls exceed anything a
small account can find -- and two numbers set those tolls: a fee coefficient of
0.07 charged to both sides, and a one-cent minimum tick that on a three-cent
contract is a third of the notional.

Polymarket US is a CFTC-regulated designated contract market listing the same
instrument, priced on the same quadratic formula, differing only in those
numbers:

|                 | Kalshi     | Polymarket US |
| --------------- | ---------- | ------------- |
| taker theta     | 0.07       | 0.06          |
| **maker theta** | **0.07**   | **-0.0125**   |
| minimum tick    | $0.01      | $0.001        |
| minimum order   | 1 contract | 1 contract    |

The maker coefficient is negative: the exchange pays for a resting quote rather
than charging for one. Surveyed across 355 live two-sided books, priced twice --
once on each schedule -- the maker round trip is positive on **all 355** here and
on 169 under Kalshi's terms. The gap is widest exactly where the volume is:

| mid band | med spread | Polymarket RT | Kalshi RT | med queue |
| -------- | ---------- | ------------- | --------- | --------- |
| 10-20c   | 100        | +124          | -190      | 16        |
| 30-40c   | 100        | +158          | -394      | 182       |
| 40-50c   | 100        | +160          | -424      | 651       |
| 60-70c   | 100        | +157          | -365      | 153       |

217 markets clear a positive round trip **and** a queue under 500 contracts.
That pair is what Kalshi never offered in a single market: there, where the
volume was the touch held a median of 124,000 contracts, and where the queue was
short nothing traded.

### The open question: does anything trade here?

Positive economics are a statement about prices. Whether anyone trades at them is
a separate question, and two measurements disagree sharply:

* 78% of open markets report zero `sharesTraded`, and the exchange's own busiest
  show 0-1,828 against open interest of 71,000 to 427,000.
* The banked books move constantly: touch size changes in 74% of observations,
  and 2,016 of 3,560 markets have moved their touch at least once.

Both can hold at once, and the combination is the worst case for a maker --
makers quoting at each other with almost no takers. That is queue competition
without flow, which is exactly what killed passive quoting on Kalshi's
thin-queue markets, and a rebate does not repair it: being paid to quote is
worth nothing if nothing ever lifts the quote.

The deciding number is not volume. Fills go in time order, so a resting order is
reached only after everything already at its price has traded. Flow divided by
queue-ahead, per interval, is the fill rate a passive quote can expect --
`scripts/polymarket_flow.py` reports it, from the cumulative counter the
collector banks beside each book. The venue publishes no trades endpoint, so
that counter differenced between passes is the only flow available.

This is the same question the Kalshi survey asked and answered badly: it measured
volume and queue separately and never divided one by the other, which is how a
board with 879 markets trading a thousand contracts a day still produced nine
round trips in seventy-two hours.

### What this is not

Evidence that market making works. It is evidence that the economics are
positive *before* adverse selection, which on Kalshi they were not. Adverse
selection is what killed passive quoting on both Kalshi instruments and it is
invisible in a snapshot: a resting order is reached only when the market trades
*through* it, a narrower and more hostile subset of the flow than any survey
sees.

Settling it needs book history, and this exchange sells none -- its candles carry
open/high/low/close/volume and no bid or ask, and that endpoint is authenticated
anyway. The Kalshi work leaned on candles that included both sides of the book,
which is what let strategies be backtested against history that already existed.
There is no such shortcut here, so `scripts/polymarket_collect.py` banks books as
they happen, into the same `quotes` table everything else reads.

## Polymarket US: measured, 2026-09-12

The open question above now has an answer for flow and for adverse selection.
Both arrived with a defect attached, which is the more useful half.

### The venue does trade into resting orders

`scripts/polymarket_adverse_selection.py --hours 48`, over 39.6h of banked books
(279,407 books, 5,744 markets): **277 prints would have reached a quote resting
at the touch -- 168 a day** -- and 198 of them traded straight through it.

`scripts/polymarket_flow.py` had put this at roughly twenty an hour, which is
about 2x too generous: it divides an interval's traded volume by *one* side's
queue, and the counter this venue publishes is two-sided. The correction is the
same shape as the Kalshi survey's, which measured volume and queue separately
and never divided one by the other.

### The platform was not there for almost any of it

Per-print accounting through the replay of 2026-09-11, 183 prints:

| why a print did not fill us | count |
| --- | --- |
| no order of ours was resting in that market | 178 |
| resting, but on the wrong side or at the wrong price | 3 |
| reachable, but the queue ahead outlasted the print | 1 |
| filled | 1 |

Neither queue position nor adverse selection. Three gates were pricing this
venue with Kalshi's fee schedule:

| gate | what it assumed | what the venue does |
| --- | --- | --- |
| `event_costing(passive=True)`, which sets the intent's expected value | ~175cc a contract at 50c | pays **-0.3cc**, a rebate |
| the risk engine's passive edge floor | the same fee | the same rebate |
| the risk engine's market-quality check | spread <= 300cc, or it is too wide to trade | a wide spread is the maker's entire product |

Against a half-spread of 50-100cc a fictitious 175cc fee makes every quote on the
venue look like a loss, and the expected-value gate then refuses it -- 47,834
refusals in a day, each for a cost that does not exist. Opening only the fee
floor changes nothing, because the same intents are refused one gate lower.
Opening only the spread requirement is worse than nothing: the strategy then
demands four cents and the risk engine refuses anything above three, so between
them they approved **zero** orders out of 34,846 intents.

### Adverse selection is real here, and it is a function of the spread

Mark-out measured from our own fill price, so the half-spread we captured is
already inside it. Positive means the quote was worth placing.

| spread at fill | n | 1m | 5m | 30m |
| --- | --- | --- | --- | --- |
| 100-200cc | 148 | -319 +- 106 | -300 +- 104 | -292 +- 109 |
| 200-400cc | 83 | -201 +- 122 | -151 +- 111 | -142 +- 148 |
| **400-1000cc** | 35 | **+155 +- 78** | **+167 +- 76** | +100 +- 163 |
| 1000cc+ | 11 | +758 +- 694 | +454 +- 620 | +376 +- 592 |

Below four cents the spread does not pay for being on the wrong side of whoever
is in a hurry. Above it, on this sample, it does: 30 of 36 such fills are
positive, across 26 markets and 16 hours, and leaving out any single market
moves the mean between +189 and +212cc.

This is why `min_spread_cc` is a strategy parameter of 400 rather than a fee
calculation. The fee arithmetic cannot see adverse selection at all, and on a
venue that pays the maker it produces a negative requirement that any spread
clears. Until it was measured, Kalshi's wrong fee had been doing this job by
accident -- demanding a wide spread for the wrong reason.

### What the corrected configuration does

Replaying 39.5h with the venue priced correctly, the four-cent floor in place and
the taker guards off a post-only order: **21 fills (13 a day), +0.1526% on the
window, mark-out +162 +- 70cc at five minutes.**

Positive, and twenty-one fills is not proof of anything.

### What is still unmeasured

A mark-out at the mid is not realised profit. Capturing it needs an exit, and
crossing a four-cent spread to take one costs more than the 167cc it is worth --
so the exit has to be passive as well, or the position has to be held to
settlement. Neither is measured yet, and that question decides whether any of
this is money.

The polling interval is the other limit: the collector sees a book about every
thirty seconds, so a move that went and came back is invisible, and the fastest
adverse moves are the ones that hurt most. That bias runs one way. These numbers
understate adverse selection, so the negative rows above are conclusive and the
positive ones are provisional.

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

2. **A cheaper venue.** Done -- see "The venue changed" above. The fill
   simulator, fee models, walk-forward splits and artifact detection all
   transferred unchanged; only the REST client and the fee coefficients were
   venue-specific.

## Reproducing any of it

```bash
.venv/Scripts/python.exe scripts/price_baseline.py --by-family
.venv/Scripts/python.exe scripts/funding_carry_check.py
.venv/Scripts/python.exe scripts/perp_basis_check.py
.venv/Scripts/python.exe scripts/perp_backtest.py --hours 336 --cache train.pkl
.venv/Scripts/python.exe scripts/perp_direction.py train.pkl test.pkl
.venv/Scripts/python.exe scripts/polymarket_adverse_selection.py --hours 48
.venv/Scripts/python.exe scripts/polymarket_backtest.py --hours 36 --report logs/polymarket_backtest.log
```

The Polymarket numbers move as the collector banks more, so pin the window with
`--end` when a figure has to be re-derived exactly:

```bash
.venv/Scripts/python.exe scripts/polymarket_adverse_selection.py --hours 48 --end 2026-09-12T19:10:00Z
```

`scripts/backfill_settled.py` banks settled-market history from Kalshi's
candlesticks — 1,014 labelled markets across 13 close dates, from history that
already existed rather than data waited for.
