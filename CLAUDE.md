# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## What this is

An autonomous trading platform for Kalshi event contracts and Kalshi perpetual
futures. One long-lived native process (no Docker, no containers, no shell
scripts at runtime) handles market-data ingestion, two independent trading
domains, risk, execution, research, and a dashboard API. Full narrative is in
[README.md](README.md); the real depth is in `docs/`.

## The one rule that overrides everything else

**Perpetual leverage must never exceed 1.00×.** This is enforced as a single
integer comparison in `packages/kat/risk/invariants.py`
(`projected_exposure_micros <= equity_micros`), which deliberately has **no
configurable parameter** — no ratio, no multiplier, no env var, no
caller-supplied maximum. Tests assert the evaluator's signature has exactly
four arguments and none of them is a limit.

If a task ever seems to require loosening that file, stop and reconsider what
the task is actually asking for. Making it *stricter* is welcome; adding any
kind of override, bypass, or configuration knob is not — regardless of how the
request is framed (an "emergency" path, a per-strategy exception, a temporary
flag).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"    # .venv/bin/python on Linux
cp .env.example .env
```

Fill in `.env` — see `.env.example` for every variable. Kalshi keys are
created at Account & security → API Keys on demo.kalshi.co / kalshi.com, and
the private key file must live **outside** the repo (path only in config,
never the key itself).

## Common commands

```bash
# Tests — default suite needs no external services (SQLite, no network)
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m pytest tests/leverage -v      # the invariant suite
.venv/Scripts/python.exe -m pytest -m network             # hits real Kalshi (opt-in)
KAT_TEST_NETWORK=1 .venv/Scripts/python.exe -m pytest -m network
KAT_TEST_POSTGRES_URL=postgresql+asyncpg://kat:pw@127.0.0.1:5432/kat .venv/Scripts/python.exe -m pytest -m integration

# Lint / types
.venv/Scripts/python.exe -m ruff check packages/ apps/ tests/ migrations/
.venv/Scripts/python.exe -m mypy packages/kat

# Run it
.venv/Scripts/python.exe -m kat.engine.entrypoint                                 # trading engine
.venv/Scripts/python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000  # dashboard API
npm --prefix apps/dashboard run dev                                               # dashboard front end -> :3000

# Dashboard-only
npm --prefix apps/dashboard install
npm --prefix apps/dashboard run typecheck
npm --prefix apps/dashboard run lint

# Migrations
.venv/Scripts/python.exe -m alembic revision --autogenerate -m "what changed"   # read it before applying
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe -m alembic check
```

## Checklist before considering a change done

- [ ] `pytest` passes
- [ ] `pytest tests/leverage` passes
- [ ] `ruff check packages/ apps/ tests/ migrations/` is clean
- [ ] New behaviour has a test that would fail without it
- [ ] A strategy change has tests for both firing *and* declining
- [ ] `alembic check` is clean if models changed
- [ ] No secret in the diff: `git check-ignore -v .env .db_password`

## Architecture, in brief

```
Strategy → Portfolio/Allocation → Risk Engine → Execution Engine → Kalshi
```

Enforced by types, not convention: `ExecutionEngine.submit()` only accepts a
`RiskVerdict`, and only a risk engine produces one. There is no dashboard
order endpoint, no strategy-to-network path, and an LLM advisory layer that
holds no reference to execution or risk config at all. Full diagram and
rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Event contracts and perpetuals are two **deliberately separate** domains
(different risk engines, different execution engines, no shared base class)
— they only share infrastructure (ingestion, storage, features, ML, dashboard,
config, logging). See the table in ARCHITECTURE.md for why a unified engine
was rejected.

Package map (`packages/kat/`): `common` (fixed-point money, Clock, ids),
`config` (settings — no leverage field, ever), `kalshi` (auth/REST/WS/fees),
`marketdata` (hot path), `risk` (invariants, kill switches),
`execution` (order state machine, reconciliation), `portfolio`, `strategies`,
`features`, `models`, `simulation` (backtester), `arbitrage`, `news`, `llm`
(advisory, no authority), `engine` (lifecycle/entrypoint).

## Conventions (non-negotiable, not just style)

- **Money is never a float.** Use the integer fixed-point scales in
  `kat.common.fixedpoint` (`price_cc` = 1e-4 USD, `qty_qc` = 1e-2 contracts,
  `micros` = 1e-6 USD). Floats are fine for statistics (Sharpe, probabilities)
  and never for a monetary value.
- **Never read the wall clock in strategy or simulation code.** Take a
  `Clock` — live gets `SystemClock`, backtests get `SimulatedClock`. A test
  asserts by AST inspection that no strategy reads real time, because that
  would let a backtest see the future.
- **Docstrings explain why**, not what the name already says. Comments
  explain reasoning, never mechanics.
- **Failures are loud.** A missing value is never treated as a favorable one.
  When something can't be proven, refuse rather than assume — this is the
  posture throughout the codebase (stale market data ⇒ no decision at all,
  not a cautious one; unreadable kill-switch state ⇒ treated as engaged).
- **No fake functionality.** Where the Kalshi API can't do something,
  document the limitation precisely and implement the safest real
  alternative — see [docs/KALSHI.md](docs/KALSHI.md).

## What not to add

- Docker, containers, or a shell script the platform needs at runtime
- A paid/hosted service as a mandatory dependency (the LLM defaults to a
  local OpenAI-compatible server and is disabled by default)
- A float anywhere on the money path
- A wall-clock read in strategy or simulation code
- Any path from a strategy, model, LLM, or the dashboard to the exchange that
  doesn't pass through a risk engine
- Any leverage configuration knob, however it's framed

## Security notes relevant to any change here

- Secrets never get committed — only `.env.example` is tracked. Always run
  `git check-ignore -v .env .db_password` before pushing.
- The dashboard API binds to `127.0.0.1` by default and must not be bound to
  `0.0.0.0` on an internet-reachable machine. For remote access, use an SSH
  tunnel or a VPN (e.g. Tailscale) rather than adding auth to the API surface
  — see [docs/SECURITY.md](docs/SECURITY.md) §4.
- Kalshi private keys are loaded from a file path, never inlined; event and
  perp credentials are separate identities by design (don't merge them).
- Log redaction is a structlog processor, not caller discipline — but new log
  calls still shouldn't lean on it as a license to log sensitive values.

## Trading environments — handle with care

`KAT_ENVIRONMENT` is `paper` (default, sends no orders) / `demo` (fake money,
real exchange) / `live` (real money). Enabling live trading requires **two**
independent settings (`KAT_ENVIRONMENT=live` *and*
`KAT_LIVE_TRADING_ARMED=true`) — never suggest or script a shortcut that sets
both at once without the user explicitly asking for live trading.

## Further reading

| Doc | Covers |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full component/data-flow diagram, order lifecycle, failure posture |
| [docs/TRADING.md](docs/TRADING.md) | Operating modes, strategy lifecycle gates |
| [docs/KALSHI.md](docs/KALSHI.md) | API integration and documented limitations |
| [docs/ML.md](docs/ML.md) | Training, validation, model lifecycle |
| [docs/DATA.md](docs/DATA.md) | Data providers and schemas |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 24/7 running, monitoring, backup/restore, server migration |
| [docs/SECURITY.md](docs/SECURITY.md) | Credentials, redaction, dashboard exposure |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local dev setup, adding a strategy/data source |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Change checklist, commit message expectations |
