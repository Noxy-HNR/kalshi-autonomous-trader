# Architecture

## The shape of the thing

One long-lived process. Inside it: market-data ingestion, two independent
trading loops, two independent risk engines, two independent execution engines,
a research loop, and the dashboard API. No containers, no supervisor script, no
subprocess for any ordinary trading operation.

```
                        ┌──────────────────────────────────┐
   Kalshi WebSocket ───►│  MarketState (in memory)         │
   (event + perp)       │  books, marks, freshness         │
                        └───────────────┬──────────────────┘
                                        │  (synchronous, microseconds)
                                        ▼
   ┌────────────┐   intents   ┌──────────────┐   sized   ┌─────────────┐
   │ Strategies │────────────►│  Portfolio   │──────────►│ Risk engine │
   └────────────┘             │  allocation  │           └──────┬──────┘
                              └──────────────┘                  │ verdict
                                                                ▼
                                                        ┌──────────────┐
                                                        │  Execution   │───► Kalshi REST
                                                        └──────┬───────┘
                                                               │
        ┌──────────────────────────────────────────────────────┘
        ▼  (asynchronous, bounded queue — never blocks the above)
   ┌─────────────┐    ┌────────────┐   ┌─────────┐   ┌───────────┐
   │ AsyncWriter │───►│ PostgreSQL │   │ Metrics │   │ Dashboard │
   └─────────────┘    └────────────┘   └─────────┘   └───────────┘
```

The single most important structural property: **everything below the dashed
line is asynchronous.** A market message updates memory and a decision follows
from memory. Persistence, analytics, dashboard updates, research, news
processing and model training all happen behind a bounded queue. There is no
database round-trip between a tick and an order.

---

## Two domains, deliberately not unified

Event contracts and perpetual futures are traded by **separate** strategies,
signal generation, portfolio management, risk engines, execution engines,
performance measurement, backtesting and lifecycle management.

They share only infrastructure: data ingestion, historical storage, feature
engineering, machine learning, research, news, arbitrage research, monitoring,
dashboard, configuration, logging, audit and model management.

### Why not one generic engine

Because the products are genuinely different, and a generic risk calculation
would have to ignore what makes them different:

| | Event contracts | Perpetuals |
| --- | --- | --- |
| Collateral | Fully collateralised | Margined |
| Maximum loss | Known exactly at entry | Depends on the mark |
| Mark price | None | Central to everything |
| Funding | None | A real cost of carry |
| Liquidation | Impossible | Possible |
| Leverage | Not a concept | **Capped at 1.00×** |
| Pricing on the wire | YES/NO bid arrays | Conventional two-sided book |
| Price format | Fixed-point dollars (legacy cents deprecated) | Fixed-point dollars only |
| Timestamps | RFC3339 and Unix seconds | Unix milliseconds (`*_ms`) |

`EventRiskEngine` and `PerpRiskEngine` share no calculation. They do not even
share a base class — the trading loop is typed loosely against the common
`evaluate(intent, state, ...) -> RiskVerdict` shape precisely so that no one is
tempted to hoist logic into a shared parent.

Kill switches are per-domain, so halting perpetuals leaves event contracts
trading and vice versa.

---

## The pipeline, and why it is enforced by types

```
Strategy → Portfolio/Allocation → Risk Engine → Execution Engine → Kalshi
```

This is not a convention. `ExecutionEngine.submit()` takes a **`RiskVerdict`**,
and there is no overload that accepts anything else. Only a risk engine produces
a `RiskVerdict`. A strategy receives a `StrategyContext` containing market state,
time-filtered features and its own position — no REST client, no execution
engine, no account balance, no route to the network.

Consequently there is no bypass for:

- an emergency flatten (it builds an `OrderIntent` and routes it through risk,
  which recognises it as exposure-reducing and approves it),
- the dashboard (the API exposes no order endpoint at all),
- a model (it produces a probability, nothing more),
- an LLM (see below),
- or a strategy that "just needs to place one order".

---

## Package map

| Package | Responsibility |
| --- | --- |
| `kat.common` | Exact fixed-point money, ids, swappable `Clock`, errors, enums |
| `kat.config` | Settings with hard clamps. **Contains no leverage field** |
| `kat.observability` | Structured logs with correlation ids; in-process metrics |
| `kat.db` | Schema, async engine, off-critical-path batched writer |
| `kat.kalshi` | Auth, rate limiting, REST + WebSocket, fee model |
| `kat.marketdata` | Order books and live market state — the hot path |
| `kat.risk` | `invariants.py`, `event_risk.py`, `perp_risk.py`, kill switches |
| `kat.execution` | Order state machine, executors, reconciliation |
| `kat.portfolio` | Capital allocation and attribution |
| `kat.strategies` | Strategy interface, shared costing, implementations |
| `kat.features` | Feature specs with availability lags, computation, pipeline |
| `kat.models` | Trainers, calibration, selection, registry |
| `kat.simulation` | Backtester, fills, costs, validation |
| `kat.arbitrage` | Logical inconsistency detection and honest pricing |
| `kat.news` | Free-first news ingestion and research |
| `kat.llm` | Advisory layer with no authority |
| `kat.engine` | Lifecycle, trading loop, ingestion, decisions, entrypoint |

---

## Data representation

All money is exact integers. No `float` touches a monetary value anywhere in the
risk path.

| Unit | Scale | Why |
| --- | --- | --- |
| `price_cc` | 1e-4 USD | Kalshi's finest tick is $0.0001, so every valid price is an exact integer |
| `qty_qc` | 1e-2 contracts | Minimum contract granularity is 0.01 |
| `micros` | 1e-6 USD | `price_cc × qty_qc` lands here exactly (1e-4 × 1e-2 = 1e-6) |

Because the product is exact, the leverage invariant reduces to an integer
comparison — `projected_exposure_micros <= equity_micros` — with no division, no
float and no epsilon. See [`fixedpoint.py`](../packages/kat/common/fixedpoint.py).

---

## The leverage invariant

`kat/risk/invariants.py` contains **no leverage parameter**. The check is that
one comparison. There is no ratio to configure, no multiplier to scale, no
environment variable, and no function signature that accepts a caller-supplied
maximum — a test asserts the evaluator's signature contains exactly four
arguments and none of them is a limit.

Before every enforcement decision, a set of behavioural canaries with known
verdicts is re-evaluated. If any layer has replaced or weakened the check, the
canaries fail and the platform raises `InvariantCompromised` and refuses to trade
rather than trading unsafely.

**Kalshi's `account_leverage` is deliberately not used.** It is
`total_position_notional / total_maintenance_margin` — with a 10% maintenance
requirement, a completely unlevered position reports 10. Using it as the
invariant would permit roughly ten times more real leverage. We compute
`notional / equity` ourselves and treat the exchange's figures as a corroborating
second opinion.

Equity is computed conservatively and its provenance recorded. Kalshi documents
`account_equity` as 0 for self-clearing members, so a zero is treated as
*absent*, not as zero dollars — and an unknown equity blocks every
exposure-increasing perpetual order.

Exposure is the **larger** of our local view and Kalshi's reported
`total_position_notional`, so a missed fill resolves in the safe direction.

---

## Order lifecycle

Our order state is ours; the exchange's is the exchange's; reconciliation is
what reconciles them.

```
CREATED ─┬─► RISK_REJECTED
         └─► SUBMITTED ─┬─► ACKNOWLEDGED ─┬─► PARTIALLY_FILLED ─► FILLED
                        │                 ├─► CANCEL_REQUESTED ─► CANCELLED
                        │                 └─► EXPIRED
                        ├─► REJECTED
                        └─► UNKNOWN  ◄── ambiguous submission
```

`UNKNOWN` is the important state. It is entered when a submission's outcome was
never observed, and it **counts fully against exposure** until reconciliation
resolves it. Resolution queries the exchange by `client_order_id` — the
idempotency key assigned before submission — and never resubmits.

`CANCEL_REQUESTED → FILLED` is legal, because a cancel races the matching engine
and can lose. A state machine that forbade it would reject the truth.

---

## Start-up sequence

Order matters, and trading is gated on the whole thing completing:

```
verify configuration → connect database → load persisted state (kill switches,
mode) → connect Kalshi → authenticate → discover capabilities → query live
account → reconcile positions/orders/fills/balances → rebuild in-memory state →
validate risk state → start ingestion → start research → health checks →
only then permit trading
```

If reconciliation cannot prove local state matches the exchange, the risk kill
switch engages and the platform runs observe-only until a human resolves it. It
never resumes anyway.

A clean shutdown writes a marker. Its absence at start-up means the previous run
died unexpectedly; that changes nothing about the sequence — reconciliation runs
identically either way — but it is recorded and surfaced.

---

## Failure posture

The rule throughout: **when the system cannot prove something is true, it refuses
to trade rather than assuming the optimistic case.**

| Failure | Response |
| --- | --- |
| Order submission times out | `UNKNOWN`, counted against exposure, never retried |
| Connect error on an order | Retried — the request never reached Kalshi |
| 429 | Retried; local bucket drained to resynchronise |
| WebSocket sequence gap | Fresh snapshot requested; deltas not applied to a book we cannot trust |
| Delta leaves book crossed | Book discarded, re-snapshot requested |
| Market data stale | No decision at all, not a cautious one |
| Account state stale | Every order rejected |
| Perp equity unknown | Exposure-increasing orders blocked; reductions permitted |
| Database unavailable | Writes queue and degrade; trading path unaffected |
| Kill switch state unreadable | Reported as **engaged** — unknown halt state is a halt |
| Process killed | Next start detects the missing clean-shutdown marker and reconciles |

---

## Latency

The critical path is short by construction. Everything expensive is off it:

- Persistence: bounded queue drained by a background task
- Analytics, dashboard, research, news, training: separate components
- Logging: `QueueHandler` drained by a background thread, so a stalled disk
  delays log persistence rather than order submission
- Connections: persistent HTTP/2 session and persistent WebSocket, so the order
  path pays no handshake

Latency is measured as tail percentiles, not means, because on an execution path
the tail is the number that matters and a mean hides it.

---

## What is deliberately absent

- **No Docker, Kubernetes, or containers.** The application is a native process.
- **No PowerShell or shell scripts at runtime.** `kat.engine.service` *generates*
  a scheduled task or systemd unit; it installs nothing.
- **No paid SaaS as a mandatory dependency.** Every dependency is free and
  open-source. The LLM defaults to a local server and is disabled by default.
- **No leverage configuration.** Stated three times in this document because it
  is the one thing that must never be added.
