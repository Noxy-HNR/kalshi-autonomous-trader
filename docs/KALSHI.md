# Kalshi API Integration

This document records how this platform integrates with Kalshi, what the current
API actually provides, and — importantly — what it does **not** provide, so that
nobody has to rediscover a limitation by hitting it in production.

The authoritative sources are vendored in `docs/vendor/kalshi/`:

| File | Contents |
| --- | --- |
| `openapi.yaml` | Trade API (event contracts), v3.28.0 |
| `perps_openapi.yaml` | Perpetuals / margin REST API |
| `asyncapi.yaml` | Event-contract WebSocket channels |
| `perps_asyncapi.yaml` | Margin WebSocket channels |
| `md/*.md` | Official guides: auth, rate limits, fees, fixed point, order direction |

Everything below was verified against those specs and, where noted, against the
live API.

---

## 1. Environments

Credentials are **not** shared between demo and production. A demo key only works
against demo hosts.

| | Production | Demo |
| --- | --- | --- |
| REST | `https://external-api.kalshi.com/trade-api/v2` | `https://external-api.demo.kalshi.co/trade-api/v2` |
| WebSocket | `wss://external-api-ws.kalshi.com/trade-api/ws/v2` | `wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2` |
| Perps REST | same host, under `/margin` | same host, under `/margin` |
| Perps WebSocket | `wss://external-api-margin-ws.kalshi.com/trade-api/ws/v2/margin` | `wss://external-api-margin-ws.demo.kalshi.co/trade-api/ws/v2/margin` |

Endpoint selection is **derived from the environment**, never configured
directly (`kat/config/settings.py`), so a demo credential cannot be pointed at
production by editing one string.

Verified live: `GET /exchange/status` returns `exchange_active: true` on both
demo and production, and market/orderbook payloads parse correctly on both.

---

## 2. Authentication

Three headers on every authenticated request:

| Header | Value |
| --- | --- |
| `KALSHI-ACCESS-KEY` | API key id |
| `KALSHI-ACCESS-TIMESTAMP` | current time in **milliseconds** |
| `KALSHI-ACCESS-SIGNATURE` | base64 RSA-PSS/SHA256 over `timestamp + METHOD + path` |

Two details cause most integration failures, and both are handled explicitly in
`kat/kalshi/auth.py`:

1. **The signed path is the full path from the API root, including the
   `/trade-api/v2` prefix, and excluding the query string.** A request to
   `https://external-api.kalshi.com/trade-api/v2/portfolio/orders?limit=5` signs
   `/trade-api/v2/portfolio/orders`.
2. **The timestamp is milliseconds, not seconds.**

The signature uses RSA-PSS with MGF1-SHA256 and a salt length equal to the digest
length. The private key is read from a file path outside the repository and is
never placed in a config value.

Separate credentials are used for event contracts and perpetuals
(`KAT_KALSHI__EVENT_CREDENTIALS__*` and `KAT_KALSHI__PERP_CREDENTIALS__*`).
Kalshi documents that keys should not be shared between the event-contract and
margin FIX gateways; separate keys also give separate rate-limit buckets and a
smaller blast radius if one is compromised.

---

## 3. Fixed-point representation

This is the most consequential detail in the whole integration.

Kalshi represents prices and quantities as **decimal strings**, not integers:

- `*_dollars` fields — fixed-point dollar strings, up to 4 decimal places for
  prices; responses emit money with up to 6.
- `*_fp` fields — fixed-point contract counts, 2 decimal places. **Fractional
  contracts are supported**, with a minimum granularity of 0.01 contracts.

The legacy integer-cent fields (`yes_price`, `count`) still exist but **cannot
represent sub-cent prices**, and markets on the finer tick grids will silently
truncate through them. This platform reads and writes the fixed-point fields
exclusively.

### Tick grids are per-market and must be read, not assumed

Each market carries `price_ranges` — an array of `{start, end, step}` bands — and
a `price_level_structure` label. **`price_ranges` is the source of truth.** The
documented structures range from `linear_cent` ($0.01 ticks) down to
`center_centi_edge_centi_cent` ($0.0001 ticks), and Kalshi adds new ones over
time.

Verified live: a demo market returned `linear_cent` (tick $0.01) while a
production multivariate market returned `deci_cent` (tick $0.001). A client that
assumed whole cents would have had its orders rejected on the second.

`EventMarket.tick_size_cc()` and `snap_price_cc()` read `price_ranges` directly.
Snapping always rounds *against* the trader — a buy limit rounds down, a sell
limit rounds up — so snapping can only ever make an order less aggressive.

### Our internal scales

| Name | Scale | Rationale |
| --- | --- | --- |
| `price_cc` | 1e-4 USD | Finest documented tick is $0.0001, so every valid price is an exact integer |
| `qty_qc` | 1e-2 contracts | Minimum contract granularity is 0.01 |
| `micros` | 1e-6 USD | Money; `price_cc * qty_qc` lands here exactly |

`notional_micros = price_cc * qty_qc` is an exact integer multiplication with no
rounding step, which is what lets the leverage invariant be an exact integer
comparison. See `kat/common/fixedpoint.py`.

---

## 4. Order books — the YES/NO reflection

**Event contracts do not publish a two-sided book.** `GET /markets/{ticker}/orderbook`
returns `orderbook_fp` containing two arrays of **bids**:

- `yes_dollars` — bids to buy YES
- `no_dollars` — bids to buy NO

Because a YES and a NO contract together always settle at exactly $1.00, a NO bid
at price `p` is economically an offer to sell YES at `1 - p`:

```
YES ask price = $1.00 - NO bid price
YES ask size  = NO bid size
```

Getting this wrong is the most common Kalshi integration bug: the resulting book
looks plausible, the spread looks plausible, and every price is inverted. The
conversion happens in exactly one place,
`OrderBook.from_event_contract_levels()`, and is covered by tests.

**Perpetuals publish a conventional two-sided book** (`bids` descending, `asks`
ascending) with no reflection.

---

## 5. Order direction

Kalshi is migrating from `side`/`action` to `outcome_side`/`book_side`:

- `outcome_side` (`yes`/`no`) — the directional exposure. `buy-yes` and `sell-no`
  both produce `yes`.
- `book_side` (`bid`/`ask`) — the same bit in book vocabulary. `bid` ≡ YES,
  `ask` ≡ NO.

The legacy `side`, `action` and `is_yes` fields are deprecated (documented as not
being removed before 2026-05-14). This platform reads the canonical fields and
falls back only where the legacy field is all that a response provides.

An important consequence: **an order at price `p` with `outcome_side=no` matches
an order at the same price `p` with `outcome_side=yes`.** Both parties trade at
the same price, in opposite directions. `outcome_side` describes exposure, not
price.

This platform only ever *buys* an outcome: to exit a YES position it buys NO
rather than selling YES. The two are economically identical on Kalshi, and
buying is the form that never depends on the exchange agreeing about inventory
we hold.

---

## 6. Rate limits

Kalshi meters in **tokens**, not request counts.

- Two independent budgets: **Read** (GETs) and **Write** (order placement,
  amends, cancels, order groups, RFQ quote flow, block trade accepts).
- **Perps are metered in their own Read and Write buckets.** There are four
  buckets in total; perp traffic never starves event-contract traffic.
- Default cost is **10 tokens**. `GET /account/endpoint_costs` is the
  authoritative list of non-default costs and is loaded at start-up.
  Known examples: `GET /margin/balance` costs 5, or **50** with
  `compute_available_balance=true`.
- Buckets refill continuously. Write buckets above Basic tier hold **two seconds**
  of budget, allowing an idle client to burst at 2× the sustained rate.
- **Batch endpoints save no tokens** — 25 orders costs 25 × 10, and the whole
  batch must fit in the bucket at once or the entire batch is rejected.
- A 429 returns `{"error": "too many requests"}` with **no `Retry-After` and no
  `X-RateLimit-*` headers**, and carries **no penalty**. The bucket keeps
  refilling.

Tier budgets (tokens/second): Basic 200/100, Advanced 300/300, Expert 600/600,
Premier 1000/1000, Paragon 2000/2000, Prime 4000/4000, Prestige 10000/8000.

`kat/kalshi/ratelimit.py` implements all four buckets with the documented burst
behaviour, and deliberately spends only a configured fraction (default 80%) of
the published budget. The headroom exists so a burst of *order* activity is never
blocked by background market-data polling draining the same bucket.

---

## 7. Fees

Fees are not a rounding detail on a sub-$250 account: a 7% quadratic fee on a
$0.50 contract is $0.0175 per contract, larger than most edges a strategy will
find.

### What the API tells us

`GET /series/{ticker}` returns `fee_type` and `fee_multiplier`. Events can
override both (`fee_type_override`, `fee_multiplier_override`; a null override
means *cleared*, not zero).

| `fee_type` | Meaning |
| --- | --- |
| `quadratic` | General Trading Fees: proportional to `C × P × (1 − P)` |
| `quadratic_with_maker_fees` | Same taker formula, maker fee at a 0.25 multiplier |
| `quadratic_with_combo_maker_fees` | Same, 0.5 maker multiplier |
| `flat` | Specific Trading Fees: flat rate per contract |

### What the API does *not* tell us — a documented limitation

**The base coefficient lives in Kalshi's published fee schedule PDF, not in the
API.** The worked examples in Kalshi's fee-rounding documentation are drawn from
markets with different multipliers and cannot be used to derive a single
constant — one example is consistent with 0.07 and another is not.

`kat/kalshi/fees.py` therefore never assumes a coefficient it was not given:

- When the series `fee_multiplier` is known, it is used and the estimate is
  marked **exact**.
- When it is unknown, a fallback of 0.07 is used, the estimate is flagged
  **inexact**, and a 1.5× safety margin is applied. The risk engine refuses to
  trade on an inexact estimate whose margin would consume the edge.

Verified: with `fee_multiplier = 0.07`, the model reproduces Kalshi's documented
worked example exactly (0.03 contracts at $0.3301 → $0.0005).

**Realised fees always come from the fill payload**, never from the model. The
model decides whether to trade; the exchange decides what was charged.

### Fee rounding

Every fill produces three components: a **trade fee** (ceiled to $0.0001), a
**rounding fee** restoring the account's target balance precision ($0.0001 for
direct members, $0.01 otherwise), and a **rebate** from accumulated rounding
overpayment. Net fee = trade + rounding − rebate, and is never negative. A
per-order accumulator makes the total across many small fills converge to what a
single equivalent fill would have cost.

---

## 8. WebSocket

Authentication is required **at handshake time** — the signed headers go in the
HTTP upgrade request — even for channels that carry only public market data.

Channels: `orderbook_delta`, `ticker`, `trade`, `fill`, `user_orders`,
`order_group_updates`, plus (event contracts only) `market_positions`,
`market_lifecycle_v2`, `multivariate_market_lifecycle`, `multivariate`,
`communications`, `cfbenchmarks_value`, `pyth_value`.

- Subscribe with `{"id": N, "cmd": "subscribe", "params": {"channels": [...]}}`.
  Use `market_ticker` (string) for one market and `market_tickers` (array) for
  several — **sending the wrong one silently subscribes to nothing.**
- `orderbook_delta` sends an `orderbook_snapshot` first, then incremental deltas.
- Every message carries a `seq`. **A gap means we missed a book update.** The
  client counts the gap, requests a fresh snapshot via
  `update_subscription` with `action: get_snapshot`, and does not continue
  applying deltas to a book it can no longer trust.
- Kalshi sends WebSocket **ping frames every 10 seconds** with body `heartbeat`;
  clients respond with pong. The `websockets` library does this automatically,
  and we additionally send our own pings so a half-open connection is detected
  within the ping timeout rather than hanging.

### Timestamp convention differs between domains

| Channel | Event contracts | Margin |
| --- | --- | --- |
| `orderbook_delta` | `ts` (RFC3339) | `ts_ms` (Unix ms) |
| `ticker` | `ts` (Unix seconds) + `time` (RFC3339) | `ts_ms` |
| `trade`, `fill` | `ts` (Unix seconds) | `ts_ms` |
| `user_orders` | `created_time` etc. (RFC3339) | `created_ts_ms` etc. |

Margin payloads use Unix milliseconds with an `_ms` suffix throughout.

---

## 9. Perpetuals (margin) specifics

"Perps", "margin" and "perpetual futures" all refer to the same product. The API
surface uses *margin* throughout.

### Leverage — the critical distinction

`GET /margin/risk` returns:

```
account_leverage = total_position_notional / total_maintenance_margin
```

**This is a margin-utilisation ratio, not equity-based leverage.** With a 10%
maintenance requirement, a completely unlevered position reports an
`account_leverage` of 10.

Using Kalshi's number as this platform's 1.00× invariant would permit roughly
**ten times** more real leverage than intended. We therefore compute our own:

```
effective_leverage = total_absolute_position_notional / account_equity
```

and use Kalshi's figures only as a corroborating second opinion. See
`kat/risk/invariants.py` and `TRADING.md`.

### Equity

`GET /margin/balance` returns per-subaccount `account_equity`,
`maintenance_margin`, `initial_margin`, `resting_orders_margin`,
`available_balance`, plus top-level `settled_funds`.

Documented gotchas, all handled in `kat/kalshi/perps/models.py`:

- `account_equity` and `initial_margin` are **0 for self-clearing members**.
- `resting_orders_margin` is **0 unless `compute_available_balance=true`**.
- `available_balance` is **0 for institutional users** or if the flag was not
  passed.

A zero in these fields is an **absence, not a value**. `equity_is_reported`
distinguishes them, and an unknown equity blocks every exposure-increasing
perpetual order rather than defaulting to anything.

### Pricing

Margin uses **decimal dollar pricing exclusively** — there are no integer-cent
fields, and the FIX `UseDollars` flag is always enabled.

### Not available on margin

Documented gaps, not omissions in this implementation:

- Batch order operations (`BatchCreateOrders`, `BatchCancelOrders`)
- Queue positions
- Events, series, milestones, multivariate collections, structured targets
- RFQs and quotes
- **Historical data endpoints**
- Exchange schedule
- WebSocket channels `market_positions`, `market_lifecycle_v2`, multivariate
- `POST /portfolio/intra_exchange_instance_transfer` (event ↔ margin transfer) is
  **not yet available**; it is enabled with the production rollout

`GET /margin/enabled` reports whether margin is on for the account in a given
environment. Perps roll out member by member, so this is checked at start-up and
the entire perpetual subsystem stays dormant when it returns false, rather than
failing every request.

### Notional risk limit

`GET /margin/notional_risk_limit` returns a per-account (and optionally
per-market) notional ceiling. It is **read-only** — there is no endpoint to set
it — so this platform treats it as an additional constraint to respect, checked
alongside its own limits.

### Order semantics

`reduce_only` is **rejected unless `time_in_force` is IOC or FOK**. The perps REST
client upgrades a reduce-only GTC order to IOC rather than letting the exchange
reject it.

`cancel_order_on_pause` defaults to true in this platform, so a resting
perpetual order does not survive an exchange pause into a market whose price has
moved — the exposure would otherwise be measured against a stale mark.

---

## 10. Order lifecycle and idempotency

`client_order_id` is the idempotency key. This platform generates it once at
order construction and **never regenerates it**.

The rule this enables: **never blindly retry a non-idempotent submission.**

`kat/kalshi/http.py` classifies every request:

| Situation | Action |
| --- | --- |
| Connect error / connect timeout | Safe to retry — the request never reached Kalshi |
| 429 | Safe to retry — documented as rejected before execution, no penalty |
| 502/503/504 on a GET | Retry with jittered backoff |
| **Read timeout on an order submission** | **`AmbiguousSubmission`** — never retried |
| **502/503/504 on an order submission** | **`AmbiguousSubmission`** — never retried |

An `AmbiguousSubmission` puts the order into internal state `UNKNOWN`, where it
**continues to count against exposure**. Reconciliation then resolves the truth
by querying the exchange for that `client_order_id` — turning an unknown outcome
into a known one rather than guessing.

---

## 11. Historical data

Event contracts expose `/historical/*` endpoints (candlesticks, fills, orders,
positions, trades, markets) plus `/historical/cutoff`. Candlesticks are also
available per market and per event.

**Perpetuals expose no historical endpoints.** Historical perp data must
therefore be accumulated by this platform's own ingestion from the live feed.
This is a real limitation with a real consequence: perp backtests can only run
over the period since ingestion began. It is documented here rather than papered
over, and `DATA.md` records the retention strategy.

---

## 12. Verified against the live API

| Check | Demo | Production |
| --- | --- | --- |
| `GET /exchange/status` | pass | pass |
| `GET /markets` parse | pass | pass |
| `GET /markets/{t}/orderbook` parse + reflection | pass | pass |
| Tick grid read from `price_ranges` | `linear_cent` (100 cc) | `deci_cent` (10 cc) |

Authenticated endpoints require credentials; see `README.md` for setup and
`tests/integration/` for the tests that exercise them (marked `network`, skipped
unless `KAT_TEST_NETWORK=1`).
