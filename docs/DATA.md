# Data

## Principles

**Two timestamps on everything ingested.** `source_ts` is when the source says it
happened; `ingested_ts` is when this process could first have acted on it. Only
the second is admissible in a backtest.

This is not pedantry. Several free government feeds back-date items, because the
timestamp they publish is the document's issue date rather than the moment it
became fetchable. A simulator filtering on `source_ts` would hand a strategy
several hours of hindsight and produce a beautiful, entirely fictitious equity
curve.

**Provenance on every record**: source, source identifier, market relationship,
both timestamps, data type, quality status.

**Synthetic data is tagged and never mixed with real history.**
`DataQuality.SYNTHETIC` and `Dataset.is_synthetic` exist so a robustness scenario
cannot be mistaken for something that happened.

---

## Kalshi data

Collected where the API provides it:

| Category | Stored as |
| --- | --- |
| Markets, events, series | `markets`, `events`, `series` |
| Prices, bid/ask, sizes | `quotes` |
| Order books | `orderbook_snapshots` |
| Public trades | `trade_prints` |
| Candles | `candles` (period discriminated) |
| Our orders and fills | `orders`, `fills` |
| Positions | `event_positions`, `perp_positions` (separate on purpose) |
| Balances and margin | `balance_snapshots` |
| Funding | `funding_rates` (realised and estimates) |
| Settlements | `settlements` |
| Market lifecycle | `market_lifecycle_events` |
| API latency and errors | `api_events` |

### Fixed point

Prices arrive as decimal strings with up to 4 decimals; money with up to 6;
contract counts with 2, and **fractional contracts are supported** down to 0.01.
Legacy integer-cent fields exist but cannot represent sub-cent prices and are not
used.

Each market's valid prices form a grid described by `price_ranges`. That is the
source of truth — tick sizes range from $0.01 down to $0.0001 depending on the
market, and Kalshi adds new structures over time. Verified live: a demo market
returned `linear_cent` (100 cc) while a production multivariate market returned
`deci_cent` (10 cc). A client that assumed whole cents would have had its orders
rejected on the second.

### The YES/NO reflection

Event contracts publish **two arrays of bids**, not a two-sided book. A NO bid at
`p` is a YES ask at `1 − p`. This conversion happens in exactly one place and is
covered by tests, because getting it wrong produces a book that looks entirely
plausible with every price inverted.

Perpetuals publish a conventional two-sided book with no reflection.

### Documented gaps

- **Perpetuals expose no historical endpoints.** Historical perp data must be
  accumulated by our own ingestion, so perp backtests can only span the period
  since ingestion began. A real limitation with a real consequence, not an
  implementation shortcut.
- **The event ↔ margin balance transfer is not yet available**, pending Kalshi's
  production rollout.
- **Batch operations, queue positions, RFQs and quotes do not exist on margin.**

---

## External data

Pluggable providers, **free sources only**. The platform never requires a paid
API.

| Category | Sources |
| --- | --- |
| News | RSS/Atom from wire services and public broadcasters |
| Government | Federal Register, agency feeds |
| Weather | NWS/NOAA public JSON |
| Economic | Public series endpoints |
| Crypto | Public exchange tickers |

Two provider classes rather than one per feed: `RSSSource` needs no configuration
beyond a URL because RSS is a fixed schema; `JSONSource` takes a small
path-mapping description, so adding a new open JSON API is a YAML entry rather
than code. Anything genuinely irregular still gets a subclass — the mapping
language is deliberately not a programming language.

All external series land in one `external_series` table keyed by provider and
series, so a new source is configuration rather than a migration. Economic data
gets revised, so `revision` lets a backtest read the value as it stood at the
time rather than the number known today.

---

## News pipeline

```
source → ingestion → normalisation → deduplication → relevance classification
  → entity/event extraction → market matching → impact analysis
  → feature generation → strategy signals
```

Deduplication is by content hash plus near-duplicate detection, so syndicated
copies of one story collapse to one article.

Entity and event extraction uses rule-based and statistical methods — regex
patterns, gazetteers, TF-IDF keyword extraction — rather than a paid NLP service.
This is honestly limited: it matches names and obvious event types reliably and
does not do deep semantic understanding.

Everything is queryable point-in-time: a backtest asking "what did we know at time
T" gets only articles with `ingested_ts <= T`.

---

## Storage

PostgreSQL, with TimescaleDB used opportunistically.

The six highest-cardinality tables (`quotes`, `orderbook_snapshots`,
`trade_prints`, `candles`, `predictions`, `api_events`) grow without bound and are
almost always queried by time range. Migration `0002` asks the server what it
supports:

- **TimescaleDB present** → hypertables, partitioned by time transparently
- **Not present** → BRIN indexes, which exploit the fact that rows are inserted in
  roughly time order. Far smaller than a B-tree and nearly as effective for
  append-only data

**TimescaleDB ships no Windows binaries**, so the desktop deployment takes the
BRIN path and the Linux server migration will take the hypertable path, with no
schema divergence between them.

### Representation

Money is `BIGINT` micro-dollars, prices `INTEGER` centicents, counts `BIGINT`
centi-contracts — identical to the in-memory representation, so values round-trip
unchanged.

Auto-incrementing primary keys use a dialect-aware type: `INTEGER` on SQLite (only
that exact spelling aliases ROWID and auto-increments) and `BIGINT` on PostgreSQL.

### Write path

The database is never on the critical path. Producers call a non-blocking
`enqueue`; a background task drains the queue in batches. Under pressure the
oldest **non-critical** rows are evicted; orders, fills, decisions, risk events and
audit rows are never dropped and are logged in full if a write is abandoned.

Rows with a natural key are merged, so a websocket replay upserts rather than
duplicates. Append-only rows are added directly, avoiding a SELECT per row on the
highest-volume write path.

---

## Retention

Market data accumulates quickly. Suggested policy, applied by an operator rather
than automatically — deleting trading history is not something a program should
decide on its own:

| Table | Suggested retention |
| --- | --- |
| `quotes` | 90 days at full resolution, then downsample to candles |
| `orderbook_snapshots` | 30 days |
| `trade_prints` | 1 year |
| `candles` | Indefinite — small, and the basis of most backtests |
| `orders`, `fills`, `decisions` | **Indefinite. This is the audit trail** |
| `audit_log`, `risk_events` | **Indefinite** |
| `api_events` | 30 days |
