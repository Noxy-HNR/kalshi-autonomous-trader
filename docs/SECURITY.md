# Security

This platform holds credentials that can move real money. The controls below are
the ones that actually matter for a single-operator system running on a desktop
and later a dedicated server.

---

## 1. What must never be committed

The `.gitignore` blocks these by pattern, but the rule is worth stating plainly:

- API keys and API key ids
- Private keys (`*.key`, `*.pem`, `*.p12`, `*.pfx`)
- Passwords and database URLs containing passwords
- Tokens of any kind
- SSH keys and certificates
- `.env` and any `configs/*.local.yaml`

Only `.env.example` is committed, and it contains placeholders.

**Verify before every push:**

```bash
git check-ignore -v .env .db_password
```

If a secret is ever committed, rotating the credential is the only real fix —
rewriting history does not help once the object has been pushed. Kalshi keys are
revocable from **Account & security → API Keys**.

---

## 2. Kalshi credentials

### Storage

The private key is read from a **file path**, never from an inline configuration
value. This is deliberate: a path in a config file is harmless if that file leaks;
an inline key is not.

Store keys outside the repository:

```
C:\secrets\kalshi\event.key      (Windows)
/etc/kat/secrets/event.key       (Linux)
```

Restrict permissions so only the account running the platform can read them:

```bash
icacls "C:\secrets\kalshi" /inheritance:r /grant:r "%USERNAME%:(OI)(CI)F"
```

On Linux:

```bash
chmod 700 /etc/kat/secrets && chmod 600 /etc/kat/secrets/*.key
```

### Separate identities per domain

Event contracts and perpetuals use separate credential blocks:

```
KAT_KALSHI__EVENT_CREDENTIALS__*
KAT_KALSHI__PERP_CREDENTIALS__*
```

Kalshi documents that keys should not be shared between the event-contract and
margin gateways. Beyond that requirement, separate keys give separate rate-limit
buckets and a smaller blast radius: a compromised event key cannot touch the
margin account.

### Environment isolation

Demo and production credentials are not interchangeable — a demo key only works
against demo hosts. The platform derives its endpoints from
`KAT_ENVIRONMENT` rather than from a configurable URL, so a demo credential
cannot be pointed at production by editing one string.

### Key handling in the process

The key is loaded once at start-up into a `cryptography` private-key object and
is never serialised, logged, or included in an exception message.
`KalshiSigner.__repr__` renders a masked key id and nothing else.

---

## 3. Log redaction

Secrets are stripped by a **structlog processor**, not by discipline at call
sites, because discipline fails and a processor does not.

Two layers:

1. **Key-name matching** — any field named `api_key`, `api_key_id`,
   `authorization`, `kalshi-access-key`, `kalshi-access-signature`, `password`,
   `private_key`, `private_key_path`, `secret`, `signature`, or `token` has its
   value replaced wholesale.
2. **Value-pattern matching** — anything that *looks* like a credential is
   redacted regardless of field name: PEM blocks and long base64 strings
   (signatures, bearer tokens).

Database URLs are masked to `scheme://****@host` before display.

`AppSettings.redacted()` produces a configuration dump safe for logs and the
dashboard, with key ids masked to `abcd...wxyz` and key paths shown as
`<configured>`.

**When adding a log call, do not rely on this.** Prefer not logging the value at
all; the redactor is a backstop, not a licence.

---

## 4. Dashboard and API exposure

The API has **no authentication**, by design. It can engage kill switches, change
the operating mode and move risk limits, and putting a login in front of that
would mean inventing a session and password story for a single-operator tool. The
decision is that **the network is the authentication**: the API answers only on
addresses that are private by construction.

`kat.net.tailscale.resolve_bind_host` enforces that. It is not advice — binding
to `0.0.0.0`, `::`, or any LAN address raises `RemoteAccessError` and the
platform does not start. There is no setting that overrides it.

### Reaching it from another device

Set one variable:

```
KAT_API_REMOTE_ACCESS=tailscale
```

The platform finds this machine's Tailscale address (`tailscale ip -4`, falling
back to interface inspection) and binds the API and the dashboard to it. Both
then answer on your tailnet and nowhere else.

Tailscale rather than a LAN address or a port-forward because a tailnet address
is a WireGuard tunnel endpoint: it exists only while the tunnel is up, only
devices you have authorised can route to it, traffic is encrypted end-to-end so
the absence of TLS on the dashboard is not the exposure it would otherwise be,
and access is revoked by removing the device from the tailnet — a control that
lives outside this process and cannot be undone by anything running in it.

Setup, once:

1. Install Tailscale on this machine **and** on the device you want to use.
2. Run `tailscale up` on both, signing into the same tailnet.
3. Set `KAT_API_REMOTE_ACCESS=tailscale` and restart.

The launcher prints the reachable URL, and `/api/system` reports
`remote_access` with `installed`, `connected` and `address` — because "I set it
up and it isn't working" otherwise looks identical to "it is working", since
both load fine on the machine they run on.

**If Tailscale is not running the platform still starts, on loopback, and logs
`remote_access.tailscale_unavailable`.** The fallback is deliberately toward
*less* exposure: a trading platform that refuses to start because a VPN is down
is worse than one reachable only from its own machine.

An SSH tunnel remains a fine alternative and needs no setting, since it
terminates on loopback:

```bash
ssh -L 8000:127.0.0.1:8000 user@server
```

### The dashboard cannot bypass safety controls

Every dashboard action that could affect trading routes through the same code as
everything else:

- Risk parameter edits are **clamped by the backend** to a safe band on load. The
  clamps are the real limits; the fields only let you be *more* conservative.
- There is no dashboard path to submit an order. The execution engine accepts
  only a `RiskVerdict`, which only a risk engine produces.
- Emergency flattening constructs an `OrderIntent` and routes it through the risk
  engine like any other order. The risk engine recognises it as exposure-reducing
  and approves it — it is not exempt from evaluation.
- **No dashboard action can raise the 1.00× perpetual leverage ceiling.** There
  is no field, endpoint, or code path that accepts a leverage value.

---

## 5. Database

- The `kat` role owns only the `kat` database and has no superuser rights.
- The password is generated at setup and stored in `.db_password`, which is
  gitignored. Treat it as a secret.
- PostgreSQL listens on `127.0.0.1` only in the default install. If you move to a
  server, use `scram-sha-256` in `pg_hba.conf` and restrict `listen_addresses`.
- Backups contain full trading history and account state. Encrypt them at rest
  and treat them with the same care as the live database — see
  [OPERATIONS.md](OPERATIONS.md).

---

## 6. LLM boundary

The LLM is an advisor with **no authority**, and this is structural rather than
stated:

- `kat.llm.advisor` holds no reference to any execution engine, REST client or
  risk configuration. It cannot construct an order.
- Its output must parse into a strict schema with a probability in `[0, 1]`;
  anything else is rejected and logged.
- Its influence on position size is capped by `settings.llm.max_influence_pct`.
- Every consultation is recorded in `llm_signals` — model, version, prompt id,
  context supplied, raw output, parsed output, accepted or rejected with the
  reason, and latency — **whether or not it was used**.

It defaults to a **local** OpenAI-compatible server (Ollama, llama.cpp, LM
Studio) and is disabled by default. The platform never requires a paid hosted
model, and never sends account balances, positions or credentials to any model.

---

## 7. Dependency and supply-chain hygiene

All dependencies are free and open-source, pinned to minimum versions in
`pyproject.toml` with a stated reason for each. No paid SaaS is a mandatory
dependency.

```bash
.venv/Scripts/python.exe -m pip list --outdated
```

CI runs lint and type checking on every push. Review dependency updates before
applying them — a compromised package in this process has access to trading
credentials.

---

## 8. Operational security

- **Run as a normal user account**, not Administrator or root. Nothing in the
  platform needs elevated privileges at runtime.
- **The kill switches are the first response to anything suspicious.** They
  persist across restarts and fail closed if their state cannot be read.
- **Audit everything material.** `audit_log` records every mode change, kill
  switch, risk parameter edit and manual intervention, with actor and timestamp.
- **Reconcile before resuming.** After any restart the platform compares its
  state against the exchange and refuses to resume trading if the two disagree in
  a way it cannot explain.

---

## 9. Reporting a problem

This is a private repository for a single operator. If you find a security defect
in the code, open an issue on the repository. Do not include credentials, key
material, or account identifiers in the report.
