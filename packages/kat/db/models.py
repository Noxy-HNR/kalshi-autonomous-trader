"""Database schema.

Organised in five groups, in dependency order:

1. **Reference** -- series, events, markets (both domains).
2. **Market data** -- quotes, book snapshots, trades, candles, funding.
3. **Trading** -- decisions, orders, fills, positions, balances.
4. **Research** -- strategies, signals, models, predictions, datasets,
   backtests, news.
5. **System** -- audit, risk events, health, API events, kill switches.

Event contracts and perpetuals share *tables* where the shape is genuinely
identical (a trade print is a trade print), discriminated by a ``market_type``
column, and use *separate tables* wherever the domains differ materially --
notably positions and balances, because perpetual margin state has no
event-contract counterpart and merging them would force exactly the generic
risk calculation this platform is required not to have.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from kat.db.base import Base, IngestMixin, TimestampMixin, utcnow

# JSONB on PostgreSQL, plain JSON on SQLite -- the same Python dict either way.
JSONType = JSON().with_variant(JSONB(), "postgresql")


# =============================================================================
# 1. Reference data
# =============================================================================


class Series(Base, TimestampMixin):
    """A Kalshi series: the recurring template behind events.

    ``fee_type`` and ``fee_multiplier`` are stored because they are the inputs
    to the fee model. We never hard-code 0.07: the multiplier comes from the
    exchange and can be overridden per event.
    """

    __tablename__ = "series"

    ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64), index=True)
    frequency: Mapped[str | None] = mapped_column(String(32))
    fee_type: Mapped[str | None] = mapped_column(String(48))
    fee_multiplier: Mapped[float | None] = mapped_column(Float)
    settlement_sources: Mapped[dict | None] = mapped_column(JSONType)
    raw: Mapped[dict | None] = mapped_column(JSONType)


class Event(Base, TimestampMixin):
    """A Kalshi event: one real-world question, holding one or more markets."""

    __tablename__ = "events"

    ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    series_ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    title: Mapped[str | None] = mapped_column(Text)
    sub_title: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64))
    mutually_exclusive: Mapped[bool | None] = mapped_column(Boolean)
    collateral_return_type: Mapped[str | None] = mapped_column(String(48))
    strike_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fee_type_override: Mapped[str | None] = mapped_column(String(48))
    fee_multiplier_override: Mapped[float | None] = mapped_column(Float)
    raw: Mapped[dict | None] = mapped_column(JSONType)


class Market(Base, TimestampMixin):
    """A tradeable instrument in either domain.

    ``market_type`` discriminates ``event`` from ``perp``. ``price_ranges`` holds
    the market's tick grid verbatim, because it is the exchange's source of
    truth for which prices are valid and we snap every order price to it rather
    than assuming whole cents.
    """

    __tablename__ = "markets"

    ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    event_ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    series_ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    title: Mapped[str | None] = mapped_column(Text)
    yes_sub_title: Mapped[str | None] = mapped_column(Text)
    no_sub_title: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(32), index=True)
    contract_kind: Mapped[str | None] = mapped_column(String(32))  # binary | scalar
    open_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    close_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    expected_expiration_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settlement_timer_seconds: Mapped[int | None] = mapped_column(Integer)
    price_level_structure: Mapped[str | None] = mapped_column(String(64))
    price_ranges: Mapped[dict | None] = mapped_column(JSONType)
    notional_value_micros: Mapped[int | None] = mapped_column(BigInteger)
    result: Mapped[str | None] = mapped_column(String(32))
    rules_primary: Mapped[str | None] = mapped_column(Text)
    last_seen_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    raw: Mapped[dict | None] = mapped_column(JSONType)

    __table_args__ = (Index("ix_markets_type_status", "market_type", "status"),)


class MarketLifecycleEvent(Base):
    """Status transitions, so a backtest knows when a market was actually open."""

    __tablename__ = "market_lifecycle_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    change: Mapped[str] = mapped_column(String(64), nullable=False)
    old_status: Mapped[str | None] = mapped_column(String(32))
    new_status: Mapped[str | None] = mapped_column(String(32))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    payload: Mapped[dict | None] = mapped_column(JSONType)


# =============================================================================
# 2. Market data
# =============================================================================


class Quote(Base, IngestMixin):
    """Top-of-book snapshot, appended on every meaningful change.

    Sized for volume: this is the highest-cardinality table in the schema and is
    a TimescaleDB hypertable when the extension is available.
    """

    __tablename__ = "quotes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bid_cc: Mapped[int | None] = mapped_column(Integer)
    ask_cc: Mapped[int | None] = mapped_column(Integer)
    bid_size_qc: Mapped[int | None] = mapped_column(BigInteger)
    ask_size_qc: Mapped[int | None] = mapped_column(BigInteger)
    last_cc: Mapped[int | None] = mapped_column(Integer)
    mark_cc: Mapped[int | None] = mapped_column(Integer)  # perps only
    volume_qc: Mapped[int | None] = mapped_column(BigInteger)
    open_interest_qc: Mapped[int | None] = mapped_column(BigInteger)

    __table_args__ = (Index("ix_quotes_ticker_ts", "ticker", "ts"),)


class OrderbookSnapshot(Base, IngestMixin):
    """Full depth at a point in time.

    Stored as JSON level arrays rather than one row per level: we read a book as
    a whole or not at all, and one row per level multiplies write volume by the
    book depth for no query benefit.
    """

    __tablename__ = "orderbook_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    seq: Mapped[int | None] = mapped_column(BigInteger)
    bids: Mapped[list | None] = mapped_column(JSONType)  # [[price_cc, size_qc], ...]
    asks: Mapped[list | None] = mapped_column(JSONType)
    depth_levels: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("ix_obs_ticker_ts", "ticker", "ts"),)


class TradePrint(Base, IngestMixin):
    """A public trade on the tape (not one of ours -- see ``Fill`` for those)."""

    __tablename__ = "trade_prints"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trade_id: Mapped[str | None] = mapped_column(String(64), index=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price_cc: Mapped[int] = mapped_column(Integer, nullable=False)
    count_qc: Mapped[int] = mapped_column(BigInteger, nullable=False)
    taker_side: Mapped[str | None] = mapped_column(String(8))

    __table_args__ = (Index("ix_trades_ticker_ts", "ticker", "ts"),)


class Candle(Base, IngestMixin):
    """OHLCV bar. ``period_s`` distinguishes 1m / 1h / 1d series in one table."""

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    period_s: Mapped[int] = mapped_column(Integer, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open_cc: Mapped[int | None] = mapped_column(Integer)
    high_cc: Mapped[int | None] = mapped_column(Integer)
    low_cc: Mapped[int | None] = mapped_column(Integer)
    close_cc: Mapped[int | None] = mapped_column(Integer)
    volume_qc: Mapped[int | None] = mapped_column(BigInteger)
    open_interest_qc: Mapped[int | None] = mapped_column(BigInteger)
    bid_close_cc: Mapped[int | None] = mapped_column(Integer)
    ask_close_cc: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("ticker", "period_s", "ts", name="uq_candle"),
        Index("ix_candles_ticker_period_ts", "ticker", "period_s", "ts"),
    )


class FundingRate(Base, IngestMixin):
    """Perpetual funding, both realised payments and forward estimates.

    Funding is a genuine cost of carry on perps and feeds directly into the
    expected-value calculation, so it is first-class data rather than a detail.
    """

    __tablename__ = "funding_rates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    rate: Mapped[float | None] = mapped_column(Float)
    next_funding_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_estimate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    payment_micros: Mapped[int | None] = mapped_column(BigInteger)

    __table_args__ = (Index("ix_funding_ticker_ts", "ticker", "ts"),)


# =============================================================================
# 3. Trading
# =============================================================================


class Decision(Base):
    """The explainability record: why one trade was attempted, or was not.

    Written for *every* considered trade, including rejections, because "why did
    it not trade?" is asked at least as often as "why did it trade?". Every field
    the requirement lists is present, plus the full leverage arithmetic for
    perps.
    """

    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    market_type: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    strategy_id: Mapped[str | None] = mapped_column(String(96), index=True)
    strategy_version: Mapped[str | None] = mapped_column(String(32))
    model_id: Mapped[str | None] = mapped_column(String(96))
    model_version: Mapped[str | None] = mapped_column(String(32))

    prediction: Mapped[float | None] = mapped_column(Float)
    market_price_cc: Mapped[int | None] = mapped_column(Integer)
    fair_value_cc: Mapped[int | None] = mapped_column(Integer)
    edge_cc: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)
    expected_value_micros: Mapped[int | None] = mapped_column(BigInteger)

    side: Mapped[str | None] = mapped_column(String(8))
    requested_qty_qc: Mapped[int | None] = mapped_column(BigInteger)
    approved_qty_qc: Mapped[int | None] = mapped_column(BigInteger)
    limit_price_cc: Mapped[int | None] = mapped_column(Integer)

    current_exposure_micros: Mapped[int | None] = mapped_column(BigInteger)
    projected_exposure_micros: Mapped[int | None] = mapped_column(BigInteger)
    estimated_fees_micros: Mapped[int | None] = mapped_column(BigInteger)
    expected_slippage_cc: Mapped[int | None] = mapped_column(Integer)

    # Perp-only leverage arithmetic, verbatim from the assessment.
    leverage_assessment: Mapped[dict | None] = mapped_column(JSONType)

    decision: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    reason_code: Mapped[str | None] = mapped_column(String(48), index=True)
    reason: Mapped[str | None] = mapped_column(Text)
    risk_checks: Mapped[dict | None] = mapped_column(JSONType)
    features: Mapped[dict | None] = mapped_column(JSONType)
    llm_signal_id: Mapped[str | None] = mapped_column(String(64))

    order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    execution_result: Mapped[dict | None] = mapped_column(JSONType)

    environment: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")
    app_version: Mapped[str | None] = mapped_column(String(32))
    git_commit: Mapped[str | None] = mapped_column(String(48))


class Order(Base, TimestampMixin):
    """Our order, in our state machine.

    ``client_order_id`` is unique and is the idempotency key: a retry reuses it,
    so a duplicate submission is rejected by the exchange rather than filling
    twice. ``exchange_order_id`` is null until the exchange acknowledges.
    """

    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    decision_id: Mapped[str | None] = mapped_column(String(64), index=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)

    market_type: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    strategy_id: Mapped[str | None] = mapped_column(String(96), index=True)

    state: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    order_type: Mapped[str] = mapped_column(String(12), nullable=False)
    book_side: Mapped[str] = mapped_column(String(8), nullable=False)
    outcome_side: Mapped[str | None] = mapped_column(String(8))
    action: Mapped[str | None] = mapped_column(String(8))
    time_in_force: Mapped[str | None] = mapped_column(String(24))
    reduce_only: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    post_only: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    limit_price_cc: Mapped[int | None] = mapped_column(Integer)
    requested_qty_qc: Mapped[int] = mapped_column(BigInteger, nullable=False)
    filled_qty_qc: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    remaining_qty_qc: Mapped[int | None] = mapped_column(BigInteger)
    avg_fill_price_cc: Mapped[int | None] = mapped_column(Integer)
    fees_micros: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    submitted_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acknowledged_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    terminal_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    submit_latency_ms: Mapped[float | None] = mapped_column(Float)
    slippage_cc: Mapped[int | None] = mapped_column(Integer)

    reject_reason: Mapped[str | None] = mapped_column(Text)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")
    state_history: Mapped[list | None] = mapped_column(JSONType)
    raw_response: Mapped[dict | None] = mapped_column(JSONType)

    __table_args__ = (Index("ix_orders_state_type", "state", "market_type"),)


class Fill(Base):
    """An execution. Keyed on the exchange's fill id so replays are idempotent."""

    __tablename__ = "fills"

    fill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    strategy_id: Mapped[str | None] = mapped_column(String(96), index=True)

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    book_side: Mapped[str | None] = mapped_column(String(8))
    outcome_side: Mapped[str | None] = mapped_column(String(8))
    price_cc: Mapped[int] = mapped_column(Integer, nullable=False)
    count_qc: Mapped[int] = mapped_column(BigInteger, nullable=False)
    is_taker: Mapped[bool | None] = mapped_column(Boolean)
    fee_micros: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")
    raw: Mapped[dict | None] = mapped_column(JSONType)


class EventPosition(Base, TimestampMixin):
    """Event-contract position. Fully collateralised: no margin fields exist.

    Kept separate from ``PerpPosition`` on purpose -- see the module docstring.
    """

    __tablename__ = "event_positions"

    ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    position_qc: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    avg_entry_cc: Mapped[int | None] = mapped_column(Integer)
    exposure_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    realized_pnl_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    unrealized_pnl_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    fees_paid_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    strategy_id: Mapped[str | None] = mapped_column(String(96), index=True)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")
    last_reconciled_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PerpPosition(Base, TimestampMixin):
    """Perpetual position, with the margin state event contracts do not have."""

    __tablename__ = "perp_positions"

    ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    subaccount: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    position_qc: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    entry_price_cc: Mapped[int | None] = mapped_column(Integer)
    mark_price_cc: Mapped[int | None] = mapped_column(Integer)
    notional_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    unrealized_pnl_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    margin_used_micros: Mapped[int | None] = mapped_column(BigInteger)
    maintenance_margin_micros: Mapped[int | None] = mapped_column(BigInteger)
    fees_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    liquidation_price_cc: Mapped[int | None] = mapped_column(Integer)
    exchange_position_leverage: Mapped[float | None] = mapped_column(Float)
    is_portfolio: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    strategy_id: Mapped[str | None] = mapped_column(String(96), index=True)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")
    last_reconciled_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BalanceSnapshot(Base):
    """Account state over time, per domain.

    ``equity_micros`` for perps is the denominator of the leverage invariant, so
    every snapshot records which source it came from and whether it was usable.
    """

    __tablename__ = "balance_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    market_type: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")

    balance_micros: Mapped[int | None] = mapped_column(BigInteger)
    portfolio_value_micros: Mapped[int | None] = mapped_column(BigInteger)
    equity_micros: Mapped[int | None] = mapped_column(BigInteger)
    available_micros: Mapped[int | None] = mapped_column(BigInteger)
    position_value_micros: Mapped[int | None] = mapped_column(BigInteger)
    maintenance_margin_micros: Mapped[int | None] = mapped_column(BigInteger)
    initial_margin_micros: Mapped[int | None] = mapped_column(BigInteger)
    resting_orders_margin_micros: Mapped[int | None] = mapped_column(BigInteger)

    total_exposure_micros: Mapped[int | None] = mapped_column(BigInteger)
    effective_leverage: Mapped[float | None] = mapped_column(Float)
    exchange_account_leverage: Mapped[float | None] = mapped_column(Float)
    equity_source: Mapped[str | None] = mapped_column(String(32))
    raw: Mapped[dict | None] = mapped_column(JSONType)


class Settlement(Base):
    """Event-contract settlement: the realised outcome of a resolved market."""

    __tablename__ = "settlements"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    settled_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    result: Mapped[str | None] = mapped_column(String(32))
    yes_count_qc: Mapped[int | None] = mapped_column(BigInteger)
    no_count_qc: Mapped[int | None] = mapped_column(BigInteger)
    revenue_micros: Mapped[int | None] = mapped_column(BigInteger)
    raw: Mapped[dict | None] = mapped_column(JSONType)

    __table_args__ = (UniqueConstraint("ticker", "settled_ts", name="uq_settlement"),)


# =============================================================================
# 4. Research
# =============================================================================


class Strategy(Base, TimestampMixin):
    """Registered strategy and its lifecycle state.

    Performance is recorded per *stage* and never conflated: ``train_metrics``,
    ``validation_metrics``, ``oos_metrics``, ``paper_metrics`` and
    ``live_metrics`` are separate columns precisely so nobody can quote a
    training Sharpe as if it were live.
    """

    __tablename__ = "strategies"

    strategy_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    family: Mapped[str | None] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    params: Mapped[dict | None] = mapped_column(JSONType)
    param_space: Mapped[dict | None] = mapped_column(JSONType)
    description: Mapped[str | None] = mapped_column(Text)

    train_metrics: Mapped[dict | None] = mapped_column(JSONType)
    validation_metrics: Mapped[dict | None] = mapped_column(JSONType)
    oos_metrics: Mapped[dict | None] = mapped_column(JSONType)
    walk_forward_metrics: Mapped[dict | None] = mapped_column(JSONType)
    robustness_metrics: Mapped[dict | None] = mapped_column(JSONType)
    paper_metrics: Mapped[dict | None] = mapped_column(JSONType)
    live_metrics: Mapped[dict | None] = mapped_column(JSONType)

    allocated_capital_micros: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    discovered_by: Mapped[str | None] = mapped_column(String(64))
    promoted_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retire_reason: Mapped[str | None] = mapped_column(Text)
    git_commit: Mapped[str | None] = mapped_column(String(48))

    __table_args__ = (UniqueConstraint("strategy_id", "version", name="uq_strategy_version"),)


class StrategyStateChange(Base):
    """Append-only lifecycle audit: who promoted what, when, on what evidence."""

    __tablename__ = "strategy_state_changes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    from_state: Mapped[str | None] = mapped_column(String(24))
    to_state: Mapped[str] = mapped_column(String(24), nullable=False)
    actor: Mapped[str] = mapped_column(String(32), nullable=False)  # system | dashboard | research
    reason: Mapped[str | None] = mapped_column(Text)
    gate_results: Mapped[dict | None] = mapped_column(JSONType)


class Signal(Base):
    """A strategy's raw opinion, before portfolio and risk have had their say."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    strategy_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    strategy_version: Mapped[str | None] = mapped_column(String(32))
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    fair_value_cc: Mapped[int | None] = mapped_column(Integer)
    edge_cc: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)
    expected_value_micros: Mapped[int | None] = mapped_column(BigInteger)
    suggested_qty_qc: Mapped[int | None] = mapped_column(BigInteger)
    regime: Mapped[str | None] = mapped_column(String(32))
    features: Mapped[dict | None] = mapped_column(JSONType)
    metadata_json: Mapped[dict | None] = mapped_column(JSONType)


class ModelRegistry(Base, TimestampMixin):
    """Model registry entry.

    Everything needed to reproduce a prediction is recorded: the dataset and
    feature versions, the exact train/validation/test windows, the metrics from
    each, and the git commit of the code that trained it.
    """

    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(48), nullable=False)  # xgboost, logistic, ...
    market_type: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    target: Mapped[str | None] = mapped_column(String(64))

    dataset_id: Mapped[str | None] = mapped_column(String(96), index=True)
    feature_set_id: Mapped[str | None] = mapped_column(String(96))
    feature_names: Mapped[list | None] = mapped_column(JSONType)
    hyperparameters: Mapped[dict | None] = mapped_column(JSONType)

    train_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    train_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    validation_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    validation_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    test_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    test_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    train_metrics: Mapped[dict | None] = mapped_column(JSONType)
    validation_metrics: Mapped[dict | None] = mapped_column(JSONType)
    test_metrics: Mapped[dict | None] = mapped_column(JSONType)
    live_metrics: Mapped[dict | None] = mapped_column(JSONType)
    calibration: Mapped[dict | None] = mapped_column(JSONType)
    feature_importance: Mapped[dict | None] = mapped_column(JSONType)

    deployment_state: Mapped[str] = mapped_column(String(24), nullable=False, default="candidate")
    artifact_path: Mapped[str | None] = mapped_column(Text)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    n_train_rows: Mapped[int | None] = mapped_column(BigInteger)
    git_commit: Mapped[str | None] = mapped_column(String(48))

    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)


class Prediction(Base):
    """One live inference, traceable back to its exact inputs.

    ``feature_ts`` is the information-time of the newest feature used. It is
    stored so that leakage can be *detected after the fact*: a prediction whose
    ``feature_ts`` is later than its own ``ts`` saw the future.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    model_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_set_id: Mapped[str | None] = mapped_column(String(96))
    feature_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    probability: Mapped[float | None] = mapped_column(Float)
    raw_output: Mapped[float | None] = mapped_column(Float)
    calibrated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    features: Mapped[dict | None] = mapped_column(JSONType)
    inference_ms: Mapped[float | None] = mapped_column(Float)
    realized_outcome: Mapped[float | None] = mapped_column(Float)  # filled in at settlement


class Dataset(Base, TimestampMixin):
    """A versioned, immutable training dataset built by the feature pipeline."""

    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    feature_set_id: Mapped[str | None] = mapped_column(String(96))
    row_count: Mapped[int | None] = mapped_column(BigInteger)
    start_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    path: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    spec: Mapped[dict | None] = mapped_column(JSONType)


class FeatureSet(Base, TimestampMixin):
    """A named, versioned set of feature definitions with availability lags."""

    __tablename__ = "feature_sets"

    feature_set_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    features: Mapped[list | None] = mapped_column(JSONType)
    # Per-feature delay between the underlying event and the moment the value
    # could actually have been computed. The simulator enforces it.
    availability_lag_ms: Mapped[dict | None] = mapped_column(JSONType)
    git_commit: Mapped[str | None] = mapped_column(String(48))


class BacktestRun(Base, TimestampMixin):
    """One backtest, out-of-sample test, or walk-forward window."""

    __tablename__ = "backtest_runs"

    run_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    strategy_version: Mapped[str | None] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    params: Mapped[dict | None] = mapped_column(JSONType)
    start_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dataset_id: Mapped[str | None] = mapped_column(String(96))
    initial_capital_micros: Mapped[int | None] = mapped_column(BigInteger)
    final_capital_micros: Mapped[int | None] = mapped_column(BigInteger)
    metrics: Mapped[dict | None] = mapped_column(JSONType)
    equity_curve: Mapped[list | None] = mapped_column(JSONType)
    trade_count: Mapped[int | None] = mapped_column(Integer)
    is_synthetic_data: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cost_model: Mapped[dict | None] = mapped_column(JSONType)
    git_commit: Mapped[str | None] = mapped_column(String(48))


class NewsArticle(Base, IngestMixin):
    """A normalised, deduplicated news item.

    ``content_hash`` deduplicates syndicated copies of the same story.
    ``published_ts`` versus ``ingested_ts`` is what stops a backtest from
    trading on a story before we could have seen it.
    """

    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    article_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    published_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    language: Mapped[str | None] = mapped_column(String(8))
    source_reliability: Mapped[float | None] = mapped_column(Float)
    entities: Mapped[list | None] = mapped_column(JSONType)
    extracted_events: Mapped[list | None] = mapped_column(JSONType)
    claims: Mapped[list | None] = mapped_column(JSONType)
    relevance: Mapped[float | None] = mapped_column(Float)


class NewsMarketLink(Base):
    """Article-to-market relevance, with the estimated impact and confidence."""

    __tablename__ = "news_market_links"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    article_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    relevance: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_impact: Mapped[float | None] = mapped_column(Float)
    direction: Mapped[str | None] = mapped_column(String(8))
    confidence: Mapped[float | None] = mapped_column(Float)
    matched_by: Mapped[str | None] = mapped_column(String(32))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (UniqueConstraint("article_id", "ticker", name="uq_news_market"),)


class LLMSignal(Base):
    """Complete record of one LLM consultation.

    Logged whether or not it influenced anything, so the question "did the model
    ever suggest this?" is answerable. The LLM cannot submit orders; this table
    records advice, not authority.
    """

    __tablename__ = "llm_signals"

    signal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    provider: Mapped[str] = mapped_column(String(48), nullable=False)
    model: Mapped[str] = mapped_column(String(96), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(48))
    prompt_id: Mapped[str | None] = mapped_column(String(64))
    market_type: Mapped[str | None] = mapped_column(String(8))
    ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    context: Mapped[dict | None] = mapped_column(JSONType)
    raw_output: Mapped[str | None] = mapped_column(Text)
    parsed_output: Mapped[dict | None] = mapped_column(JSONType)
    probability: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    accepted: Mapped[bool | None] = mapped_column(Boolean)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    decision_id: Mapped[str | None] = mapped_column(String(64), index=True)
    latency_ms: Mapped[float | None] = mapped_column(Float)


class ArbitrageOpportunity(Base):
    """A detected pricing inconsistency and its full cost breakdown.

    Recorded whether or not it was traded, because the distribution of edges we
    *declined* is the evidence that the cost model is calibrated.
    """

    __tablename__ = "arbitrage_opportunities"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    opportunity_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    kind: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False)
    legs: Mapped[list | None] = mapped_column(JSONType)
    gross_edge_micros: Mapped[int | None] = mapped_column(BigInteger)
    fees_micros: Mapped[int | None] = mapped_column(BigInteger)
    spread_cost_micros: Mapped[int | None] = mapped_column(BigInteger)
    slippage_micros: Mapped[int | None] = mapped_column(BigInteger)
    net_edge_micros: Mapped[int | None] = mapped_column(BigInteger)
    execution_probability: Mapped[float | None] = mapped_column(Float)
    latency_risk_micros: Mapped[int | None] = mapped_column(BigInteger)
    residual_exposure_micros: Mapped[int | None] = mapped_column(BigInteger)
    worst_case_micros: Mapped[int | None] = mapped_column(BigInteger)
    expected_value_micros: Mapped[int | None] = mapped_column(BigInteger)
    actioned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)


class RegimeObservation(Base):
    """Detected market regime over a window, per market or globally."""

    __tablename__ = "regime_observations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    scope: Mapped[str] = mapped_column(String(24), nullable=False)  # global | market
    ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    market_type: Mapped[str | None] = mapped_column(String(8))
    regime: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    features: Mapped[dict | None] = mapped_column(JSONType)


# =============================================================================
# 5. System
# =============================================================================


class AuditLog(Base):
    """Append-only record of every material action and control change."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subject: Mapped[str | None] = mapped_column(String(128), index=True)
    detail: Mapped[dict | None] = mapped_column(JSONType)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    environment: Mapped[str | None] = mapped_column(String(8))


class RiskEvent(Base):
    """A risk limit that fired. Leverage blocks land here with full arithmetic."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    decision_id: Mapped[str | None] = mapped_column(String(64), index=True)
    market_type: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    strategy_id: Mapped[str | None] = mapped_column(String(96))
    reason_code: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    message: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="warning")
    limit_name: Mapped[str | None] = mapped_column(String(64))
    limit_value: Mapped[str | None] = mapped_column(String(64))
    observed_value: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[dict | None] = mapped_column(JSONType)


class KillSwitchState(Base, TimestampMixin):
    """Persisted halt state. Survives restarts by construction.

    A kill switch that forgets it was engaged when the process restarts is worse
    than none at all, so this is a table rather than a flag in memory.
    """

    __tablename__ = "kill_switch_state"

    scope: Mapped[str] = mapped_column(String(24), primary_key=True)
    subject: Mapped[str] = mapped_column(String(128), primary_key=True, default="")
    engaged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reason: Mapped[str | None] = mapped_column(Text)
    engaged_by: Mapped[str | None] = mapped_column(String(32))
    engaged_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    released_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class HealthEvent(Base):
    """Component health transitions, for the System tab and post-mortems."""

    __tablename__ = "health_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    component: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[dict | None] = mapped_column(JSONType)


class ApiEvent(Base):
    """Every non-trivial exchange interaction, for latency and error analysis."""

    __tablename__ = "api_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    domain: Mapped[str] = mapped_column(String(8), nullable=False)
    method: Mapped[str] = mapped_column(String(8), nullable=False)
    path: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    status: Mapped[int | None] = mapped_column(Integer, index=True)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class ErrorEvent(Base):
    """Unhandled or notable exceptions with enough context to reproduce."""

    __tablename__ = "error_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    component: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    error_type: Mapped[str] = mapped_column(String(96), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    traceback: Mapped[str | None] = mapped_column(Text)
    context: Mapped[dict | None] = mapped_column(JSONType)


class SystemState(Base, TimestampMixin):
    """Small key/value store for state that must survive a restart.

    Used for the operating mode, the armed flag, sequence checkpoints and the
    last clean-shutdown marker that crash recovery keys off.
    """

    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(96), primary_key=True)
    value: Mapped[dict | None] = mapped_column(JSONType)
    updated_by: Mapped[str | None] = mapped_column(String(32))


class StrategyPerformance(Base):
    """Rolling performance per strategy per stage, recomputed on a schedule."""

    __tablename__ = "strategy_performance"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    strategy_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    window: Mapped[str] = mapped_column(String(24), nullable=False)
    trades: Mapped[int | None] = mapped_column(Integer)
    win_rate: Mapped[float | None] = mapped_column(Float)
    expectancy_micros: Mapped[int | None] = mapped_column(BigInteger)
    pnl_micros: Mapped[int | None] = mapped_column(BigInteger)
    sharpe: Mapped[float | None] = mapped_column(Float)
    sortino: Mapped[float | None] = mapped_column(Float)
    max_drawdown_pct: Mapped[float | None] = mapped_column(Float)
    avg_slippage_cc: Mapped[float | None] = mapped_column(Float)
    calibration_error: Mapped[float | None] = mapped_column(Float)
    confidence_interval: Mapped[dict | None] = mapped_column(JSONType)
    degradation_flags: Mapped[list | None] = mapped_column(JSONType)


class ExternalSeries(Base, IngestMixin):
    """Generic external time series (economic, weather, crypto, sports, ...).

    One table with a provider/series key rather than a table per provider, so a
    new free data source is a config entry and not a migration.
    """

    __tablename__ = "external_series"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    series_key: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    text_value: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JSONType)
    # Economic data gets revised. ``revision`` lets a backtest read the value as
    # it stood at the time rather than the number we know today.
    revision: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (
        Index("ix_external_provider_key_ts", "provider", "series_key", "ts"),
        UniqueConstraint("provider", "series_key", "ts", "revision", name="uq_external_point"),
    )


# Tables that benefit from TimescaleDB hypertables when the extension exists.
# Migrations consult this list; on plain PostgreSQL they get BRIN indexes instead.
HYPERTABLES: tuple[tuple[str, str], ...] = (
    ("quotes", "ts"),
    ("orderbook_snapshots", "ts"),
    ("trade_prints", "ts"),
    ("candles", "ts"),
    ("predictions", "ts"),
    ("api_events", "ts"),
)

__all__ = [
    "HYPERTABLES",
    "ApiEvent",
    "ArbitrageOpportunity",
    "AuditLog",
    "BacktestRun",
    "BalanceSnapshot",
    "Candle",
    "Dataset",
    "Decision",
    "ErrorEvent",
    "Event",
    "EventPosition",
    "ExternalSeries",
    "FeatureSet",
    "Fill",
    "FundingRate",
    "HealthEvent",
    "KillSwitchState",
    "LLMSignal",
    "Market",
    "MarketLifecycleEvent",
    "ModelRegistry",
    "NewsArticle",
    "NewsMarketLink",
    "Order",
    "OrderbookSnapshot",
    "PerpPosition",
    "Prediction",
    "Quote",
    "RegimeObservation",
    "RiskEvent",
    "Series",
    "Settlement",
    "Signal",
    "Strategy",
    "StrategyPerformance",
    "StrategyStateChange",
    "SystemState",
    "TradePrint",
]

# Silence unused-import warnings for types used only in annotations.
_ = (Numeric, ForeignKey)
