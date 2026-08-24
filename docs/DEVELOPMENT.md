# Development

## Setup

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

On Linux the interpreter is `.venv/bin/python`.

The package is installed in editable mode so `kat` and `apps` import from
anywhere, not only from the repository root.

---

## Running the tests

```bash
.venv/Scripts/python.exe -m pytest
```

The default suite needs **no external services**: SQLite stands in for PostgreSQL
and nothing touches the network. It should pass on a clean checkout immediately.

Selected suites:

```bash
.venv/Scripts/python.exe -m pytest tests/leverage -v
```

```bash
.venv/Scripts/python.exe -m pytest tests/failure -v
```

Against real PostgreSQL:

```bash
KAT_TEST_POSTGRES_URL=postgresql+asyncpg://kat:pw@127.0.0.1:5432/kat .venv/Scripts/python.exe -m pytest -m integration
```

Against the real Kalshi API (skipped by default):

```bash
KAT_TEST_NETWORK=1 .venv/Scripts/python.exe -m pytest -m network
```

---

## Linting and types

```bash
.venv/Scripts/python.exe -m ruff check packages/ apps/ tests/ migrations/
```

```bash
.venv/Scripts/python.exe -m mypy packages/kat
```

Ruff config is in `pyproject.toml`. `UP042` is ignored deliberately:
`class X(str, Enum)` is what lets the enums round-trip through pydantic and
SQLAlchemy string columns unchanged.

---

## Dashboard

```bash
npm --prefix apps/dashboard install
```

```bash
npm --prefix apps/dashboard run dev
```

```bash
npm --prefix apps/dashboard run typecheck
```

The dev server proxies `/api/*` to the engine, so the engine's origin is never
baked into the client bundle. The destination is built from the **same**
variables the engine itself reads -- `KAT_API_HOST` and `KAT_API_PORT` -- so a
changed port takes effect on both sides from one setting. `KAT_API_URL`
overrides it outright, for the case where the engine genuinely runs elsewhere.

Proxying is done by a route handler (`app/api/[...path]/route.ts`) rather than a
`rewrites()` entry, because Next.js resolves rewrites at **build** time and
freezes them into the route manifest -- which would make the engine's origin a
property of whoever ran `next build`. The handler reads the environment on each
request, so the variables need only be present where `next start` runs.

---

## Migrations

After changing a model:

```bash
.venv/Scripts/python.exe -m alembic revision --autogenerate -m "what changed"
```

Read the generated file before applying it. Autogenerate is good at columns and
indexes and poor at anything requiring judgement.

```bash
.venv/Scripts/python.exe -m alembic upgrade head
```

Verify the schema still matches the models:

```bash
.venv/Scripts/python.exe -m alembic check
```

The URL comes from application settings, not `alembic.ini`, so the two cannot
disagree.

---

## Conventions

**Money is never a float.** Use the integer scales in `kat.common.fixedpoint`.
Floats are fine for statistics — Sharpe, probabilities, correlations — and never
for a monetary value.

**Never read the wall clock in strategy or simulation code.** Take a `Clock`. Live
passes `SystemClock`, backtests pass `SimulatedClock`. A test asserts by AST
inspection that no strategy reads real time, because that would let it see the
future during a backtest.

**Docstrings explain why.** The module docstring should say what non-obvious
decision the module encodes, not restate its name. Comments explain reasoning,
never mechanics.

**Failures are loud.** A missing value is not a favourable one. When something
cannot be proven, refuse rather than assume.

**No fake functionality.** If something cannot be implemented, implement the
safest real alternative and document the limitation precisely.

---

## Adding a strategy

1. Subclass `TradingStrategy` in `kat/strategies/{event,perp}/`.
2. Declare a **bounded** `param_space` — an unbounded search space is how an
   optimiser discovers a position nobody would take deliberately.
3. Declare `requires_features` and an honest `warmup_ms`.
4. Compute expected value net of a **round trip** using `kat.kalshi.fees`.
5. Use `self.skip(reason)` when declining, so "it did nothing" always comes with
   "because".
6. Add a test that it fires in the situation it targets **and** declines when the
   edge is absent. A strategy that never trades and one that is broken look
   identical from the outside.

New strategies start at `DEVELOPMENT` and climb the ladder. Nothing reaches live
capital by being registered.

---

## Adding a data source

1. Add a `SourceSpec` to `configs/news_sources.yaml` for RSS or a mapped JSON API.
2. For something irregular, subclass `NewsSource`.
3. Record both `published_ts` and `ingested_ts`. Never let a backtest filter on
   the first.

---

## Working on the leverage invariant

Do not, unless the change makes it stricter.

`kat/risk/invariants.py` deliberately contains no leverage parameter. Tests assert
that no settings field anywhere contains the word "leverage" and that the
evaluator's signature has exactly four arguments, none of which is a limit. The
behavioural canaries run before every enforcement call.

If you are editing that file to make something pass, stop and reconsider what you
are making pass.

---

## Repository hygiene

Before pushing:

```bash
git check-ignore -v .env .db_password
```

CI fails the build if key material or a `.env` is ever tracked.
