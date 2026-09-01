# Operations

## Running continuously

The platform is one native process. It handles its own start-up, shutdown,
reconnection, health checks, recovery and graceful termination. Nothing supervises
it beyond restarting the process if it exits.

```bash
.venv/Scripts/python.exe -m kat.engine.entrypoint
```

To generate a startup entry:

```bash
.venv/Scripts/python.exe -m kat.engine.service
```

That writes `deploy/kat.service`, `deploy/kat-task.xml` and a README, and prints
the registration commands. It **installs nothing** — registering a startup entry
modifies the machine and is your call.

### Windows

A Scheduled Task at logon that **restarts the process if it dies** (every minute,
up to 999 times), at LIMITED integrity. Nothing needs elevation, and a trading
process running as Administrator enlarges the blast radius for no benefit.

Registered from XML rather than `schtasks` switches, because restart-on-failure
cannot be expressed as a switch and three of the task engine's defaults would
otherwise stop a long-running process on their own: it halts tasks after three
days, stops them when the machine goes on battery, and stops them when the
machine stops being idle. The generated XML overrides all four.

```
schtasks /Create /TN "KalshiAutonomousTrader" /XML "deploy\kat-task.xml" /F
```

### Knowing it went down

Every start reports how long the platform was not running. The health loop writes
a heartbeat every 30 seconds, so a start after an unclean stop logs
`app.previous_shutdown_was_unclean` with `downtime_s` — the difference between
knowing it restarted and knowing nobody was watching the positions for five
hours. When the previous run left no heartbeat the field is `null`, not `0`:
an unknown gap is not a proven-absent one.

### Linux

A systemd unit with `Restart=always` and `TimeoutStopSec=45`. The generous stop
timeout matters: the application drains its write queue and records a
clean-shutdown marker on the way out. Killed sooner, the next start correctly
treats it as a crash.

### A reboot does not arm live trading

The process reads its configuration at start. Real orders require **both**
`KAT_ENVIRONMENT=live` and `KAT_LIVE_TRADING_ARMED=true`. A rebooted machine comes
back in whatever mode its configuration says.

---

## Monitoring

The dashboard's System tab shows process and host resources, queue depths, latency
as **tail percentiles**, market-data freshness, component health, and whether
remote access is connected.

The trading loop also logs a `trading.summary` line every 60 seconds with what it
examined and, when nothing traded, why each strategy declined:

```
trading.summary cycles=60 markets_examined=60 markets_skipped_stale=0
  intents_generated=0 declined_because={"realized_vol_unavailable": 7080,
  "no_model_probability": 3540, "too_far_from_expiry": 3422}
```

That is the first place to look when the platform is up and not trading. The
tallies come from the strategies themselves and are also on `/api/strategies`
under each strategy's `health`.

Health from the command line:

```bash
curl -s http://127.0.0.1:8000/api/health
```

### What to watch

| Signal | Concerning when | Meaning |
| --- | --- | --- |
| `trading_permitted` | false unexpectedly | A kill switch, a degraded component, or a failed reconciliation |
| `phase` | `degraded` | A critical component is unhealthy; observing but not trading |
| Write queue depth | Growing steadily | The database cannot keep up, or is down |
| Dropped rows | Non-zero | Persistence is losing data under pressure |
| `risk.leverage_blocks` | Rising | Strategies repeatedly asking for more than 1.00× |
| Sequence gaps | Rising | Websocket messages are being missed |
| Market data age | Above the freshness budget | The feed has stalled; nothing will trade |
| `unknown` orders | Any | A submission outcome was never observed; reconciliation will resolve it |

---

## Logs

Structured JSON, rotating at 64 MB with 10 backups, in `logs/`.

Every record carries a correlation id, so one market tick, the decision it
produced, the order that followed and the fill that came back share a key:

```bash
grep '"correlation_id":"cor_abc123"' logs/kat.log
```

File writes go through a background thread. A stalled disk delays log persistence,
never order submission.

Secrets are stripped by a processor rather than by discipline at call sites: known
credential field names are replaced wholesale, and anything that *looks* like a
credential — a PEM block, a long base64 string — is redacted regardless of field
name.

---

## Backups

### What must survive

| Data | Why |
| --- | --- |
| `orders`, `fills`, `decisions` | The audit trail. Irreplaceable |
| `audit_log`, `risk_events` | What the system did and why |
| `strategies`, `strategy_state_changes` | Lifecycle history and evidence |
| `model_registry` + `data/models/` | Deployed models and their provenance |
| Market data | Replaceable in principle, expensive in practice |

### Schedule

Daily full dump, retained 30 days:

```bash
pg_dump -U kat -h 127.0.0.1 -Fc kat > backups/kat-$(date +%F).dump
```

Model artefacts alongside it — a registry row whose artefact is missing is a model
you cannot load:

```bash
tar -czf backups/models-$(date +%F).tar.gz data/models/
```

Backups contain full trading history and account state. **Encrypt them at rest**
and treat them with the same care as the live database.

### Restore

```bash
pg_restore -U kat -h 127.0.0.1 -d kat --clean --if-exists backups/kat-2026-08-24.dump
```

```bash
.venv/Scripts/python.exe -m alembic upgrade head
```

Then start the platform. It reconciles against the exchange before trading and
refuses to trade if the two disagree.

### Test the restore

A backup that has never been restored is a hypothesis. Restore into a scratch
database periodically and confirm `alembic check` passes against it.

---

## Recovery

### After an unclean shutdown

Nothing special is required. The start-up sequence is identical either way:
reconciliation queries the exchange and adopts its view. The missing
clean-shutdown marker is recorded and surfaced, and raises the bar for re-arming
live trading.

### After a reconciliation failure

The risk kill switch engages automatically and the platform runs observe-only.
Investigate before releasing it:

```bash
curl -s http://127.0.0.1:8000/api/risk/events
```

Look for `position_mismatch`, `phantom_position` or `equity_unavailable`. Compare
against the exchange directly. Release the switch only once you understand the
discrepancy — it exists precisely to stop the platform trading through a state it
cannot explain.

### Orphaned orders

Orders resting at the exchange that this process is not tracking, typically
survivors of a crash before persistence. Reconciliation reports them and
`cancel_orphans` cancels them: an untracked resting order can fill into a position
nothing is managing.

---

## Server migration

The same source code runs on Linux. Nothing in the trading logic is
Windows-specific.

1. **Provision** PostgreSQL 17 with the TimescaleDB extension available. The
   migrations detect it and create hypertables instead of BRIN indexes; the schema
   is otherwise identical.
2. **Move secrets** to `/etc/kat/kat.env`, outside the repository, read by systemd.
3. **Restore** the database dump and run `alembic upgrade head`. Migration `0002`
   takes the hypertable path on the new server.
4. **Copy** `data/models/` so registry rows still resolve to artefacts.
5. **Install** the systemd unit from `deploy/`.
6. **Start in paper mode** and confirm reconciliation succeeds before arming
   anything.

Keep application, configuration, secrets, data and infrastructure separate. The
repository holds only the first.

---

## Routine checks

**Daily** — dashboard Overview: equity, drawdown, trading permitted, component
health. Any `unknown` orders.

**Weekly** — strategy performance per stage, and anything degrading. Model
calibration drift. Backup completed and restorable.

**Monthly** — dependency updates. Log and database growth against disk. Restore a
backup into a scratch database.

---

## Going live

Arming real money is four settings and a restart. Everything below is about
whether you should, and how to tell afterwards.

### Before you arm

The platform will happily trade real money the moment it is armed, so these are
your checks, not its.

- [ ] **A paper track record you believe.** Not "it ran without crashing" -- a
      period long enough to contain losing days, with positions opened *and
      closed*, and equity above where it started at the end of it. The equity
      curve is in `balance_snapshots`; the closed trades are in `fills`.
- [ ] **A deployed model whose gates you have read.** `/api/models` shows what is
      deployed. The gates it cleared are recorded on the row: test AUC, excess
      calibration error, and the train-to-test gap. A model that has not cleared
      them is not deployed, and forcing one is not a supported operation.
- [ ] **`alembic check` is clean** and a backup has been restored into a scratch
      database at least once. See Backups above.
- [ ] **The private key lives outside the repository**, and `git check-ignore -v
      .env` says the environment file is ignored.
- [ ] **Position and loss limits sized for the account you are actually risking.**
      The shipped defaults assume a sub-$250 account: 10% of equity per market,
      60% total exposure, 2% loss per trade, 5% daily loss, 20% drawdown. On a
      larger account these are not conservative, they are merely small.
- [ ] **You have decided what a bad day looks like** and set
      `KAT_EVENT_RISK__MAX_DAILY_LOSS_PCT` to it. The platform halts there. If you
      would not accept that loss, the number is wrong.

### Arming

```bash
# in .env
KAT_ENVIRONMENT=live
KAT_LIVE_TRADING_ARMED=true
KAT_OPERATING_MODE=supervised
KAT_KALSHI__PROD_CREDENTIALS__API_KEY_ID=...
KAT_KALSHI__PROD_CREDENTIALS__PRIVATE_KEY_PATH=C:/secrets/kalshi/prod.key
```

Then restart the engine. Both `environment=live` and `live_trading_armed=true`
are required, separately and deliberately: neither alone will place an order,
and the dashboard can set neither.

### Immediately after

```bash
curl -s http://127.0.0.1:8000/api/overview | jq '{environment, live_armed, can_submit_real_orders, trading_permitted}'
```

`can_submit_real_orders` must be `true` and `equity_is_simulated` must be
**absent or false**. If equity still reads as simulated, the platform is running
on a seeded balance rather than your account, and every size it calculates is
wrong.

Watch the first fills against the exchange's own record. The fee on a live fill
comes from Kalshi's payload, not from the model, so a disagreement between
`fills.fee_micros` and your Kalshi statement is a real problem and worth stopping
for.

### Starting smaller than you intend to finish

`KAT_OPERATING_MODE=supervised` requires approval before promotion between
strategy stages, and probation caps a newly promoted strategy at
`research.probation_capital_pct` of equity. Using both means the first real money
at risk is a fraction of the account rather than all of it.

Deposit what you are prepared to lose rather than sizing the limits down against
a larger balance. A percentage limit on a bigger deposit is a bigger loss.

---

## Emergency

**Stop everything immediately** — from the dashboard Controls tab, or:

```bash
curl -X POST http://127.0.0.1:8000/api/control/pause
```

**Cancel every working order:**

```bash
curl -X POST http://127.0.0.1:8000/api/control/cancel-all
```

**Disarm real money**: set `KAT_LIVE_TRADING_ARMED=false` and restart. The
dashboard cannot arm or disarm live trading; that is deliberate.

Kill switches persist across restarts, so a halt survives a reboot. If you are
unsure whether the platform is safe to run, the safe action is to leave the switch
engaged: it costs opportunity, and the alternative costs money.
