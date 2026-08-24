# Trading

## Objective

The platform optimises for **survival and compounding**, not activity.

It is designed to be comfortable taking no trade. Most candidate trades are
declined, and that is the intended behaviour rather than a failure of the signal
generation. What it optimises for:

- risk-adjusted return
- expected value **after real costs**
- robustness across regimes
- manageable drawdown
- capital efficiency on a small account

What it explicitly does **not** optimise for: number of trades, activity,
leverage, or raw return without regard to risk.

---

## The cost reality that shapes everything

Kalshi's quadratic fee is proportional to `price × (1 − price)`, so a round trip
costs about **3.5 cents per contract near $0.50** and about **1.3 cents near the
extremes**.

| Price | One-way fee | Round trip |
| --- | --- | --- |
| $0.10 | $0.0063 | ~$0.013 |
| $0.25 | $0.0132 | ~$0.026 |
| $0.41 | $0.0170 | ~$0.034 |
| $0.50 | $0.0175 | ~$0.035 |
| $0.90 | $0.0063 | ~$0.013 |

*(at a 0.07 series multiplier; the actual multiplier comes from the API)*

The consequence is concrete and drives strategy behaviour: **a three-cent edge is
a losing trade in the middle of the range and a good one at the tails.** The
mean-reversion strategy fires on a four-cent reversion at $0.10 and declines the
identical signal at $0.41, and that is correct, not conservative.

Every strategy computes expected value net of a **round trip**, not just the
entry. Counting only the entry fee roughly doubles an apparent edge, which is the
most common way a backtest lies.

---

## Operating modes

| Mode | The system decides | You decide |
| --- | --- | --- |
| **A** Fully Autonomous | Everything: markets, direction, probability, EV, entry, exit, timing, size, strategy, order type, allocation | Nothing |
| **B** Autonomous Execution | Market selection, timing, entries, exits, sizing, execution, portfolio, risk | Which strategies are permitted |
| **C** Autonomous Discovery | As A, plus discovering, optimising, validating and deploying new strategies | Nothing |
| **D** Gated Discovery *(default)* | As C, but nothing reaches live capital without clearing every gate | Nothing, but nothing is trusted either |

Changing mode never changes a risk limit.

---

## Strategy lifecycle

```
DISCOVERY → DEVELOPMENT → BACKTEST → OUT-OF-SAMPLE → WALK-FORWARD
  → ROBUSTNESS → PAPER → PROBATION → LIVE → MONITORING → REVIEW → RETIRED
```

Promotion moves **one rung at a time** and only when that rung's gate passes.
Demotion to `PAUSED` or `RETIRED` is always allowed. `next_promotion_state()`
returns `None` for states outside the ladder, so a paused or retired strategy
cannot be promoted back by accident.

### Gates

| Gate | Requirement |
| --- | --- |
| Backtest | Positive expectancy after realistic costs |
| Out-of-sample | Sharpe ≥ `min_oos_sharpe`, drawdown ≤ `max_oos_drawdown_pct` |
| Walk-forward | ≥ `min_walk_forward_windows` rolling windows, consistent sign |
| Robustness | Survives Monte Carlo resampling and ±10% parameter perturbation |
| Sample size | ≥ `min_trades_for_promotion` |
| Degradation | In-sample-to-out-of-sample Sharpe decay ≤ `max_is_oos_sharpe_degradation` |
| Paper | ≥ `min_paper_trades_for_live` with performance consistent with out-of-sample |
| Probation | Trades live at `probation_capital_pct` of normal allocation |

Performance is reported **per stage and never blended**. A backtest Sharpe and a
live Sharpe are different claims about the world; the dashboard gives each its
own column and shows a dash — not zero — for a stage that has not run.

---

## Strategies

### Event contracts

| Strategy | Trades when | Stands aside when |
| --- | --- | --- |
| `mispricing` | Model probability diverges from price beyond costs | EV after fees is not positive |
| `mean_reversion` | A move is large vs the market's own volatility, **unconfirmed by volume**, and decelerating | Volume confirms it (that is news, not overreaction), or it is still extending |
| `momentum` | Volume-confirmed continuation | Volume is ordinary — that is one order walking a thin book |
| `expiry_drift` | Market is a clear favourite near resolution | Neither side is a favourite, or liquidity is thin |
| `news_reaction` | A fresh, reliable, high-impact story is matched to the market | The story is stale past an explicit cutoff |

`mean_reversion` and `momentum` are **mutually exclusive by construction**: one
requires volume below a threshold, the other above it. They cannot both fire on
the same move, which is the honest relationship between the two ideas.

### Perpetuals

| Strategy | Trades when |
| --- | --- |
| `funding_carry` | The funding rate is large enough to pay for the spread, taking the receiving side, with a hard cap on holding period |
| `trend` | A trend in the mark price persists |
| `basis` | Mark and index diverge, with a stale-index guard that produces no order rather than a wrong one |

Every perpetual strategy is subject to the 1.00× ceiling and will be resized by
the risk engine without complaint.

---

## Risk

### Event contracts

Fully collateralised: maximum loss is known exactly at entry (`qty × price`), so
sizing needs no volatility estimate to bound the downside. The binding
constraints are capital, concentration and loss budgets.

| Limit | Default | Clamped to |
| --- | --- | --- |
| Position size | 10% of equity | ≤ 25% |
| Portfolio exposure | 60% | ≤ 95% |
| Loss per trade | 2% | ≤ 10% |
| Daily loss | 5% | ≤ 25% |
| Drawdown | 20% | ≤ 50% |
| Correlated exposure (one event) | 25% | ≤ 60% |
| Open positions | 20 | ≤ 500 |
| Order notional | $25 | ≤ $10,000 |

**Correlated exposure** treats every market within one Kalshi event as a single
bet, because they resolve off the same real-world outcome. Holding five markets
in one event is one position, not five.

### Perpetuals

Everything above, plus margin, funding, marks and liquidation — and the
invariant.

> **Perpetual leverage never exceeds 1.00×.**

`max_exposure_pct_equity` can only make the platform *more* conservative. It is
clamped to ≤ 1.00, and the invariant is applied independently afterwards
regardless of what it says.

### The invariant in detail

```
effective_leverage = total_absolute_position_notional / account_equity ≤ 1.00
```

Implemented as the integer comparison `projected_exposure_micros <=
equity_micros`. Exactly 1.0000× is permitted; 1.0001× is refused by one part in
ten thousand.

**Both directions consume exposure.** A short of 10 contracts carries the same
notional risk as the equivalent long; shorts are not netted against longs across
markets.

**Working orders consume headroom.** An order resting on the book is an order
that can fill, so its full notional counts.

**De-risking is permitted and is not a bypass.** A mark-price move can push
leverage above 1.00× with no order of ours involved. Orders that *strictly
reduce* exposure remain permitted, because forbidding the only orders that fix
the situation would be the unsafe choice. An order claiming to reduce exposure
that would in fact increase it is rejected like any other.

**Unknown equity blocks everything that increases exposure.** Reductions are
still allowed, because flattening into an unknown state lowers risk.

---

## Kill switches

Five independent scopes: `GLOBAL`, `EVENT`, `PERP`, `STRATEGY`, `RISK`.

- **They persist.** State lives in the database and reloads at start-up. A kill
  switch that forgets it was engaged is worse than none, because it creates false
  confidence.
- **They fail closed.** If state cannot be read, the switch reports *engaged*.
- **They work when the database is broken.** Engaging flips in-memory state
  immediately and queues the write — which is exactly the failure that makes you
  want one.
- `RISK` engages automatically when reconciliation fails or a serious risk
  condition trips.

---

## Execution

- Market and limit orders, cancellation, amendment, partial fills, expiration
- Internal order state machine independent of the exchange's
- Slippage and latency tracked per order
- **Never assume an order succeeded because an HTTP request succeeded**

Order prices are snapped to the market's own tick grid, read from `price_ranges`
rather than assumed to be whole cents. Snapping always rounds *against* the
trader — a buy limit down, a sell limit up — so it can only make an order less
aggressive.

This platform only ever **buys** an outcome. To exit a YES position it buys NO
rather than selling YES. The two are economically identical on Kalshi, and buying
never depends on the exchange agreeing about inventory we believe we hold.

---

## Capital allocation

Not equal weight. Allocation is proportional to a **shrunk, correlation-penalised
edge estimate** drawn from out-of-sample or later performance only. A backtest
buys nothing.

- **Shrinkage by sample size** (`n / (n + k)`): a 12-trade Sharpe of 3.0 loses to
  a 400-trade Sharpe of 1.2, because the first is mostly luck.
- **Correlation penalty**: capital in a strategy that moves with the rest of the
  book is worth less at the margin.
- **Fractional Kelly**, well below one, because full Kelly is optimal only for a
  perfectly known edge.
- **Degraded strategies get nothing**, with the reason recorded.
- **Probation** gets `probation_capital_pct` of what it would otherwise receive.

### The small-account case

With $40 deployable and five eligible strategies, dividing produces five
positions that cannot pay their own round trip. Below a stated floor the
allocator **concentrates** — funding in score order until the money runs out —
and above it diversifies. The switch is a visible threshold because it is the
opposite of what a large-account allocator should do.

Event and perpetual capital are allocated separately from separate pools.

---

## Live trading gates

Before real orders are permitted:

- [ ] All tests pass (`pytest`)
- [ ] Leverage tests pass (`pytest tests/leverage`)
- [ ] Failure-injection tests pass (`pytest tests/failure`)
- [ ] Reconciliation works against the real account
- [ ] Paper trading has run and produced sensible decisions
- [ ] `alembic check` reports no schema drift
- [ ] Restart and recovery verified
- [ ] Kill switches verified engaged and released
- [ ] Strategy validation gates working
- [ ] Secrets outside the repository, `.env` gitignored
- [ ] Dashboard reachable and showing live state
- [ ] Logs written and rotating
- [ ] Monitoring showing component health

Then, deliberately:

```bash
KAT_ENVIRONMENT=live
```

```bash
KAT_LIVE_TRADING_ARMED=true
```

Two independent switches. Setting `live_trading_armed` without
`environment=live` is a **configuration error at load time**, not a surprise at
the first order. Neither can be set from the dashboard: arming real money stays a
deliberate change made at the machine.

When real orders can be submitted the dashboard shows `ENVIRONMENT: LIVE` with a
red pulsing border and explicit "REAL MONEY AT RISK" text.

---

## What the platform will not do

- Trade on stale data. Stale produces *no* decision, not a cautious one.
- Resume after a failed reconciliation.
- Retry an order whose outcome it does not know.
- Exceed 1.00× perpetual leverage under any circumstance, from any code path.
- Treat a missing value as a favourable one.
- Report a training result as if it were a live one.
