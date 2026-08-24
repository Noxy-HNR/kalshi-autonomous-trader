# Kalshi Autonomous Trader

An autonomous trading platform for **Kalshi event contracts** and **Kalshi
perpetual futures**, built to run continuously on a Windows desktop and migrate
unchanged to a Linux server.

The two domains are traded by **independent** strategies, signal generation,
portfolio management, risk engines, execution engines, performance measurement,
backtesting, paper trading and lifecycle management. They share only
infrastructure: data ingestion, historical storage, feature engineering, machine
learning, research, news, arbitrage research, monitoring, dashboard,
configuration, logging, audit and model management.

---

## The one rule that is not negotiable

> **Perpetual leverage never exceeds 1.00×.**

This is an architectural invariant, not a setting. It is enforced in code, tested
exhaustively, displayed on the dashboard, logged on every rejection, and cannot
be reached from a strategy, a model, an LLM, the dashboard, configuration, or an
emergency path.

**How it is enforced.** `packages/kat/risk/invariants.py` contains no leverage
parameter at all. The check is the integer comparison

```
projected_exposure_micros <= equity_micros
```

There is no ratio to configure, no multiplier to scale, no environment variable
to read, and no function signature that accepts a caller-supplied maximum. To
raise the limit you would have to edit that comparison — which is the point.

Because all money is exact integers in micro-dollars (see
[`fixedpoint.py`](packages/kat/common/fixedpoint.py)), 1.0000× passes and
1.0001× fails by exactly one part in ten thousand, with no floating-point
ambiguity anywhere.

**Behavioural canaries.** Before every enforcement decision, a set of known
inputs with known verdicts is re-evaluated. If any layer has replaced or weakened
the check — a plugin, a monkeypatch, a careless edit — the canaries fail and the
platform refuses to trade rather than trading unsafely.

**Kalshi's own number is deliberately not used.** `GET /margin/risk` returns
`account_leverage = total_position_notional / total_maintenance_margin`, a
margin-utilisation ratio. With a 10% maintenance requirement, a completely
unlevered position reports `account_leverage` of 10. Using it as this platform's
invariant would permit roughly ten times more real leverage. We compute
`notional / equity` ourselves and use the exchange's figures only as a
corroborating second opinion. See [`docs/KALSHI.md`](docs/KALSHI.md).

**Reducing exposure is not a bypass.** A mark-price move can push leverage over
the line with no order of ours involved. Orders that *strictly reduce* exposure
remain permitted, because forbidding the only orders that fix the situation would
be the unsafe choice. An order claiming to reduce exposure that would in fact
increase it is rejected like any other.

See [`tests/leverage/test_invariant.py`](tests/leverage/test_invariant.py) — 62
tests covering the boundary, malformed input, missing equity, tamper detection,
and the absence of any configuration route to a higher limit.

---

## Requirements

| Component | Version | Notes |
| --- | --- | --- |
| Python | 3.11+ | Developed and tested on 3.13 |
| PostgreSQL | 14+ | 17 recommended. TimescaleDB optional |
| Node.js | 20+ | Dashboard only |
| OS | Windows 10/11 or Linux | No Docker, no PowerShell, no shell scripts required at runtime |

---

## Installation

```bash
git clone https://github.com/Noxy-HNR/kalshi-autonomous-trader.git
cd kalshi-autonomous-trader
python -m venv .venv
```

Then install dependencies with the venv's interpreter:

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

On Linux the interpreter is `.venv/bin/python` instead.

### Database

Create the database and role, then run migrations:

```bash
psql -U postgres -c "CREATE ROLE kat LOGIN PASSWORD 'choose-a-strong-password';"
```

```bash
psql -U postgres -c "CREATE DATABASE kat OWNER kat ENCODING 'UTF8';"
```

```bash
.venv/Scripts/python.exe -m alembic upgrade head
```

TimescaleDB is used opportunistically: migrations create hypertables when the
extension is present and plain tables with BRIN time indexes when it is not, so
the same schema works either way. **TimescaleDB does not currently ship Windows
binaries**, so a Windows deployment runs on plain PostgreSQL; the Linux server
migration picks up hypertables automatically.

---

## Configuration

Copy `.env.example` to `.env` and fill it in. `.env` is gitignored.

```bash
cp .env.example .env
```

Settings resolve in this order, highest priority first:

1. Real environment variables
2. `.env`
3. `configs/default.yaml`
4. Model defaults

YAML sits *below* the environment deliberately: the YAML file is committed and
holds shared defaults, while the environment holds machine-specific values and
secrets. If YAML won, a committed default would silently override your real
database URL.

### Kalshi credentials

Create API keys at **Account & security → API Keys** on
[demo.kalshi.co](https://demo.kalshi.co) or [kalshi.com](https://kalshi.com).
Credentials are **not** shared between demo and production.

Save the downloaded `.key` file **outside this repository** — the private key
cannot be retrieved after you close the page.

```bash
KAT_KALSHI__EVENT_CREDENTIALS__API_KEY_ID=your-key-id
KAT_KALSHI__EVENT_CREDENTIALS__PRIVATE_KEY_PATH=C:/secrets/kalshi/event.key
KAT_KALSHI__PERP_CREDENTIALS__API_KEY_ID=your-perp-key-id
KAT_KALSHI__PERP_CREDENTIALS__PRIVATE_KEY_PATH=C:/secrets/kalshi/perp.key
```

Separate identities for the two domains are recommended: Kalshi documents that
keys should not be shared between the event-contract and margin gateways, and
separate keys give separate rate-limit buckets and a smaller blast radius if one
is compromised.

---

## Launching

The trading engine is a native long-lived process. No Docker, no containers, no
shell script, no subprocesses for ordinary trading operations.

```bash
.venv/Scripts/python.exe -m kat.engine.entrypoint
```

The dashboard API:

```bash
.venv/Scripts/python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

The dashboard front end:

```bash
npm --prefix apps/dashboard run dev
```

Then open **http://127.0.0.1:3000**.

For 24/7 operation as a Windows service or a Linux systemd unit, see
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

### What the first launch does

The first start **never** trades real money. It:

1. initialises the application and verifies configuration
2. initialises the database and checks migrations are current
3. connects to Kalshi and verifies authentication
4. determines supported API capabilities (including whether perps are enabled
   for your account — they roll out member by member)
5. begins market-data ingestion
6. initialises research infrastructure
7. starts the dashboard API
8. runs health checks
9. **remains in paper mode**

---

## Operating modes

Selected from the dashboard. All four are implemented.

| Mode | Name | What the system decides |
| --- | --- | --- |
| **A** | Fully Autonomous | Markets, direction, probability, EV, entry, exit, timing, size, strategy, order type, allocation |
| **B** | Autonomous Execution | You choose permitted strategies; the system handles everything else |
| **C** | Autonomous Discovery | The system discovers, optimises, backtests, validates, paper trades and deploys strategies |
| **D** | **Gated Discovery** (default) | As C, but nothing reaches live capital without clearing every lifecycle gate |

Mode D is the recommended default. A newly discovered strategy must pass through:

```
DISCOVERY → DEVELOPMENT → BACKTEST → OUT-OF-SAMPLE → WALK-FORWARD
  → ROBUSTNESS → PAPER → PROBATION → LIVE → MONITORING → REVIEW → RETIRED
```

Promotion moves one rung at a time and only when that rung's gate passes.
Demotion to `PAUSED` or `RETIRED` is always allowed.

---

## Trading environments

| `KAT_ENVIRONMENT` | Behaviour |
| --- | --- |
| `paper` | Simulates fills against **live** market data. Sends **no** orders. |
| `demo` | Sends real orders to Kalshi's demo exchange (fake money). |
| `live` | Sends real orders with **real money**. |

### Enabling live trading

Two independent switches are required, so no single stray value can arm real
money:

```bash
KAT_ENVIRONMENT=live
KAT_LIVE_TRADING_ARMED=true
```

Setting `live_trading_armed=true` without `environment=live` is a **configuration
error at load time**, not a surprise at the first order.

Before live trading is permitted, the gates in
[`docs/TRADING.md`](docs/TRADING.md) must pass: all tests green, leverage tests
green, reconciliation working, paper trading working, migrations clean,
restart/recovery verified, kill switches verified, secrets secure, dashboard
working, logging and monitoring working.

When real orders can be submitted, the dashboard displays **`ENVIRONMENT: LIVE`**
prominently.

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest
```

The default suite requires no external services — SQLite stands in for
PostgreSQL and nothing touches the network.

Tests that hit the real Kalshi API are marked `network` and skipped unless
enabled:

```bash
KAT_TEST_NETWORK=1 .venv/Scripts/python.exe -m pytest -m network
```

The leverage suite alone:

```bash
.venv/Scripts/python.exe -m pytest tests/leverage -v
```

Linting and type checking:

```bash
.venv/Scripts/python.exe -m ruff check packages/ tests/ apps/
```

---

## Running a backtest

```bash
.venv/Scripts/python.exe -m apps.backtester --strategy mispricing --start 2026-01-01 --end 2026-06-30
```

The simulator enforces information-time integrity: a simulation at time `T` can
never see data that became available after `T`. This is a hard guarantee with a
leakage detector, not a convention — see [`docs/ML.md`](docs/ML.md).

Training, validation, out-of-sample, paper and live performance are reported
**separately** and never blended.

---

## Repository layout

```
kalshi-autonomous-trader/
├── apps/
│   ├── api/              FastAPI backend for the dashboard
│   ├── dashboard/        Next.js + TypeScript front end
│   ├── trader/           Native trading engine entry point
│   ├── data_ingestion/   Market-data and external-data workers
│   ├── research/         Autonomous research loop
│   ├── backtester/       Backtest CLI
│   └── trainer/          Model training jobs
├── packages/kat/
│   ├── common/           Exact fixed-point money, ids, clocks, errors, enums
│   ├── config/           Settings (no leverage field exists here)
│   ├── observability/    Structured logging, metrics
│   ├── db/               Schema, async engine, off-path batched writer
│   ├── kalshi/           REST + WebSocket clients, auth, rate limits, fees
│   │   ├── event_contracts/
│   │   └── perps/
│   ├── marketdata/       In-memory books and market state (the hot path)
│   ├── risk/             invariants.py, event_risk.py, perp_risk.py, kill switches
│   ├── execution/        Order state machine, executors, reconciliation
│   ├── portfolio/        Capital allocation and attribution
│   ├── strategies/       Strategy interface and implementations
│   ├── features/         Feature specs with availability lags, computation
│   ├── models/           Model registry, trainers, calibration, inference
│   ├── simulation/       Backtester, fills, costs, validation
│   ├── discovery/        Strategy discovery, optimisation, lifecycle gates
│   ├── arbitrage/        Kalshi-native arbitrage detection
│   ├── news/             News ingestion and research pipeline
│   ├── llm/              Advisory LLM layer (no order authority)
│   ├── research/         Regime detection, research loop
│   └── engine/           Application lifecycle and orchestration
├── tests/                unit, integration, leverage, simulation, failure, e2e
├── migrations/           Alembic
├── configs/              default.yaml, news_sources.yaml
├── docs/                 ARCHITECTURE, TRADING, KALSHI, ML, DATA, OPERATIONS, SECURITY, DEVELOPMENT
└── .github/workflows/    Native CI (no Docker)
```

---

## Documentation

| Document | Contents |
| --- | --- |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components and data flow |
| [TRADING.md](docs/TRADING.md) | Modes, strategies, risk, execution, live gates |
| [KALSHI.md](docs/KALSHI.md) | API integration and its documented limitations |
| [ML.md](docs/ML.md) | Training, validation, model lifecycle, deployment |
| [DATA.md](docs/DATA.md) | Data providers and schemas |
| [OPERATIONS.md](docs/OPERATIONS.md) | 24/7 running, monitoring, backups, recovery, migration |
| [SECURITY.md](docs/SECURITY.md) | Credentials and operational security |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local development and testing |

---

## Design commitments

**No fake functionality.** Nothing fabricates API responses, historical data,
model training or backtest results. Where the Kalshi API does not support
something, the limitation is documented precisely and the safest practical
alternative is implemented. Known gaps are listed in `docs/KALSHI.md`.

**The database is never on the critical path.** Market data arrives, state
updates in memory, a decision is made, risk validates it and the order goes out.
Persistence, analytics, dashboard updates, research, news processing and model
training all happen asynchronously behind a bounded queue.

**The objective is survival and compounding, not activity.** The system is
designed to be comfortable taking no trade. It optimises for risk-adjusted
return, expected value after real costs, robustness and manageable drawdown —
never for trade count, and never for leverage.

**It starts small.** The platform is built to operate correctly with an account
under $250, including fractional contracts, and to scale as equity grows. No
component hard-codes an assumption that requires a large account.
