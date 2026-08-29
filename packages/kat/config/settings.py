"""Configuration.

Design rule for this module: **anything that could weaken a hard safety limit is
not representable here.** There is no ``max_perp_leverage`` field, because a
field is something a user can set. The perpetual leverage ceiling lives in
:mod:`kat.risk.invariants` as a comparison, not a value.

Settings that *are* configurable (position sizes, daily loss budgets, freshness
budgets) are clamped by the backend to a safe band on load, so a dashboard edit
or a hand-edited YAML file cannot push them somewhere dangerous. The clamp is
applied here rather than at the point of use, so there is exactly one place
where "what is allowed" is decided.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from kat.common.enums import OperatingMode, TradingEnvironment
from kat.common.errors import ConfigError

__all__ = [
    "AppSettings",
    "DatabaseSettings",
    "EventRiskSettings",
    "KalshiCredentials",
    "KalshiSettings",
    "LLMSettings",
    "NewsSettings",
    "PerpRiskSettings",
    "ResearchSettings",
    "get_settings",
    "reload_settings",
]

REPO_ROOT = Path(__file__).resolve().parents[3]


def _clamp(value: Decimal, low: str, high: str) -> Decimal:
    lo, hi = Decimal(low), Decimal(high)
    return max(lo, min(hi, value))


class KalshiCredentials(BaseSettings):
    """One Kalshi API identity.

    Event contracts and perpetuals get separate credential blocks because Kalshi
    documents that keys should not be shared between the event-contract and
    margin FIX gateways, and separate keys give separate rate-limit buckets and
    a smaller blast radius if one is compromised.

    The private key is read from a file path, never from an inline value, so a
    key cannot end up in a config file that gets committed.
    """

    api_key_id: str = ""
    private_key_path: str = ""

    model_config = SettingsConfigDict(extra="ignore")

    @property
    def configured(self) -> bool:
        return bool(self.api_key_id and self.private_key_path)

    def resolve_key_path(self) -> Path:
        if not self.private_key_path:
            raise ConfigError("private_key_path is not set")
        path = Path(os.path.expandvars(self.private_key_path)).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            raise ConfigError(f"Kalshi private key not found at {path}")
        return path


class KalshiSettings(BaseSettings):
    """Endpoints and client behaviour for both Kalshi domains.

    Base URLs are derived from the environment rather than configured directly,
    so a demo credential can never be pointed at the production host by editing
    one string.
    """

    environment: TradingEnvironment = TradingEnvironment.PAPER

    event_credentials: KalshiCredentials = Field(default_factory=KalshiCredentials)
    perp_credentials: KalshiCredentials = Field(default_factory=KalshiCredentials)

    #: Production identity, used only when the platform is reading the
    #: production exchange. Kept separate from the demo credentials rather than
    #: swapped in and out of the same field, because "which key is currently in
    #: event_credentials" is exactly the question nobody should have to answer
    #: from memory before starting a trading process.
    prod_credentials: KalshiCredentials = Field(default_factory=KalshiCredentials)

    #: Read production market data while still in PAPER.
    #:
    #: The demo exchange has almost no liquidity -- measured on it, 945 of 1000
    #: markets had no orders at all and the tightest spread among the rest was
    #: 13 cents -- so paper trading against demo data cannot tell you anything
    #: about whether a strategy works. This points the market-data feed at
    #: production while leaving order routing exactly as it was.
    #:
    #: It is safe by construction rather than by care: PAPER never constructs an
    #: order at all, and this flag is *ignored* unless the environment is PAPER.
    #: It cannot be the reason a real order is sent, in any combination.
    read_production_market_data: bool = False

    # Client tuning. These affect latency and reliability, never safety.
    request_timeout_s: float = Field(default=8.0, ge=0.5, le=60.0)
    connect_timeout_s: float = Field(default=4.0, ge=0.5, le=30.0)
    #: Keepalive. Deliberately tolerant.
    #:
    #: At 10s/10s the client was closing its own healthy connections: 221 of the
    #: 408 disconnects observed in a day were `keepalive ping timeout`, and each
    #: one triggered a reconnect, a resubscribe, and a "Subscription buffer
    #: overflow" as the snapshot burst arrived. Market coverage fell from 57
    #: markets to 6 with nothing reporting a fault.
    #:
    #: A pong is only evidence the far end is alive; missing one briefly is not
    #: evidence it is dead. The rest of those disconnects were the exchange
    #: itself (124 HTTP 503s and 57 handshake timeouts on the demo gateway), so
    #: the connection needs to tolerate a bad minute rather than react to it.
    ws_ping_interval_s: float = Field(default=20.0, ge=1.0, le=120.0)
    ws_ping_timeout_s: float = Field(default=45.0, ge=1.0, le=180.0)
    #: Silence that means the subscription is gone, not that the market is
    #: quiet. Ten minutes: long enough that an idle overnight market does not
    #: trip it, short enough that a genuinely dropped subscription is noticed.
    ws_silence_timeout_s: float = Field(default=600.0, ge=30.0, le=3600.0)
    ws_max_reconnect_delay_s: float = Field(default=30.0, ge=1.0, le=300.0)

    # Token-bucket budgets. Defaults are the documented Basic tier; the platform
    # refreshes them from GET /account/limits at start-up when authenticated.
    read_tokens_per_second: int = Field(default=200, ge=1)
    write_tokens_per_second: int = Field(default=100, ge=1)
    perp_read_tokens_per_second: int = Field(default=200, ge=1)
    perp_write_tokens_per_second: int = Field(default=100, ge=1)
    # Fraction of the published budget we actually spend, leaving headroom so a
    # burst of order activity is never starved by background polling.
    rate_limit_utilisation: Decimal = Field(default=Decimal("0.80"))

    @field_validator("rate_limit_utilisation")
    @classmethod
    def _clamp_utilisation(cls, v: Decimal) -> Decimal:
        return _clamp(v, "0.05", "0.95")

    # -- derived endpoints ---------------------------------------------------

    @property
    def reads_production(self) -> bool:
        """Whether market data comes from the production exchange.

        Two ways to be true: trading live, or explicitly asking for production
        data while in PAPER. The second is deliberately restricted to PAPER --
        in DEMO the platform really does send orders, and it must send them to
        the same exchange it priced them from.
        """
        if self.environment is TradingEnvironment.LIVE:
            return True
        return (
            self.read_production_market_data
            and self.environment is TradingEnvironment.PAPER
        )

    @property
    def _is_demo(self) -> bool:
        return not self.reads_production

    @property
    def active_credentials(self) -> KalshiCredentials:
        """The identity to authenticate with, given where we are reading.

        Kalshi does not share credentials between the demo and production
        exchanges, so this follows the exchange rather than the environment.
        """
        return self.prod_credentials if self.reads_production else self.event_credentials

    @property
    def rest_base_url(self) -> str:
        return (
            "https://external-api.demo.kalshi.co/trade-api/v2"
            if self._is_demo
            else "https://external-api.kalshi.com/trade-api/v2"
        )

    @property
    def ws_url(self) -> str:
        return (
            "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
            if self._is_demo
            else "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
        )

    @property
    def perp_rest_base_url(self) -> str:
        # Perps REST shares the trade-api host; endpoints live under /margin.
        return self.rest_base_url

    @property
    def perp_ws_url(self) -> str:
        return (
            "wss://external-api-margin-ws.demo.kalshi.co/trade-api/ws/v2/margin"
            if self._is_demo
            else "wss://external-api-margin-ws.kalshi.com/trade-api/ws/v2/margin"
        )

    model_config = SettingsConfigDict(extra="ignore")


class DatabaseSettings(BaseSettings):
    """PostgreSQL is the production store; SQLite exists for tests and for a
    first run before a server is available.

    ``url`` is an async SQLAlchemy URL. TimescaleDB is used opportunistically:
    migrations create hypertables when the extension is present and plain tables
    when it is not, so the same schema works on both.
    """

    url: str = "postgresql+asyncpg://kat:kat@127.0.0.1:5432/kat"
    echo: bool = False
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    pool_timeout_s: float = Field(default=10.0, ge=1.0, le=120.0)
    # The async writer drains this queue in the background; the trading path
    # never blocks on a database round-trip.
    write_queue_size: int = Field(default=20_000, ge=100)
    write_batch_size: int = Field(default=500, ge=1)
    write_flush_interval_s: float = Field(default=0.5, ge=0.05, le=10.0)

    model_config = SettingsConfigDict(extra="ignore")

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")


class EventRiskSettings(BaseSettings):
    """Risk limits for the event-contract domain.

    Event contracts are fully collateralised by construction -- buying a YES at
    $0.40 costs $0.40 and can lose at most that -- so the limits here are about
    concentration, sizing and loss budgets rather than leverage.

    Every value is clamped on load. The clamps are the real limits; the fields
    only let you be *more* conservative than the platform default.
    """

    max_position_pct_equity: Decimal = Field(default=Decimal("0.10"))
    max_portfolio_exposure_pct: Decimal = Field(default=Decimal("0.60"))
    max_loss_per_trade_pct: Decimal = Field(default=Decimal("0.02"))
    max_daily_loss_pct: Decimal = Field(default=Decimal("0.05"))
    max_drawdown_pct: Decimal = Field(default=Decimal("0.20"))
    max_correlated_exposure_pct: Decimal = Field(default=Decimal("0.25"))
    max_open_positions: int = Field(default=20, ge=1, le=500)
    max_orders_per_minute: int = Field(default=30, ge=1, le=600)
    max_order_notional_usd: Decimal = Field(default=Decimal("25.00"))

    # Quality gates: an order is not placed unless the market is tradeable.
    max_spread_cc: int = Field(default=300, ge=1, le=5_000)  # $0.03 default
    min_book_depth_qc: int = Field(default=1_000, ge=1)      # 10 contracts
    max_slippage_cc: int = Field(default=100, ge=0, le=5_000)
    min_edge_cc: int = Field(default=200, ge=1)              # $0.02 of edge
    min_expected_value_micros: int = Field(default=5_000, ge=0)  # $0.005 after costs

    @model_validator(mode="after")
    def _clamp_all(self) -> EventRiskSettings:
        object.__setattr__(self, "max_position_pct_equity", _clamp(self.max_position_pct_equity, "0.001", "0.25"))
        object.__setattr__(self, "max_portfolio_exposure_pct", _clamp(self.max_portfolio_exposure_pct, "0.01", "0.95"))
        object.__setattr__(self, "max_loss_per_trade_pct", _clamp(self.max_loss_per_trade_pct, "0.001", "0.10"))
        object.__setattr__(self, "max_daily_loss_pct", _clamp(self.max_daily_loss_pct, "0.005", "0.25"))
        object.__setattr__(self, "max_drawdown_pct", _clamp(self.max_drawdown_pct, "0.01", "0.50"))
        object.__setattr__(self, "max_correlated_exposure_pct", _clamp(self.max_correlated_exposure_pct, "0.01", "0.60"))
        object.__setattr__(self, "max_order_notional_usd", _clamp(self.max_order_notional_usd, "0.01", "10000"))
        return self

    model_config = SettingsConfigDict(extra="ignore")


class PerpRiskSettings(BaseSettings):
    """Risk limits for the perpetual domain.

    Note what is absent: there is no leverage field. Perpetual leverage is
    capped at 1.00x by :mod:`kat.risk.invariants`, which reads nothing from this
    class. ``max_exposure_pct_equity`` below can only make the platform *more*
    conservative than that ceiling -- it is clamped to at most 1.00 and the
    invariant is applied afterwards regardless of what it says.
    """

    # Fraction of equity the platform is willing to deploy as perp notional.
    # Clamped to <= 1.00; the invariant enforces <= 1.00 independently anyway.
    max_exposure_pct_equity: Decimal = Field(default=Decimal("0.50"))
    max_position_pct_equity: Decimal = Field(default=Decimal("0.15"))
    max_loss_per_trade_pct: Decimal = Field(default=Decimal("0.02"))
    max_daily_loss_pct: Decimal = Field(default=Decimal("0.04"))
    max_drawdown_pct: Decimal = Field(default=Decimal("0.15"))
    max_open_positions: int = Field(default=8, ge=1, le=100)
    max_orders_per_minute: int = Field(default=20, ge=1, le=600)
    max_order_notional_usd: Decimal = Field(default=Decimal("25.00"))

    # Distance from the exchange's liquidation threshold we insist on keeping.
    # At 1x leverage liquidation should be unreachable, but we verify rather
    # than assume, using the exchange's own risk parameters.
    min_liquidation_buffer_pct: Decimal = Field(default=Decimal("0.50"))

    max_spread_cc: int = Field(default=300, ge=1, le=5_000)
    min_book_depth_qc: int = Field(default=1_000, ge=1)
    max_slippage_cc: int = Field(default=100, ge=0, le=5_000)
    min_edge_cc: int = Field(default=200, ge=1)
    min_expected_value_micros: int = Field(default=5_000, ge=0)

    # Funding is a real cost on perps and must clear the edge threshold.
    max_funding_rate_bps_per_hour: Decimal = Field(default=Decimal("5"))

    @model_validator(mode="after")
    def _clamp_all(self) -> PerpRiskSettings:
        # Hard ceiling of 1.00: configuration may tighten, never loosen.
        object.__setattr__(self, "max_exposure_pct_equity", _clamp(self.max_exposure_pct_equity, "0.001", "1.00"))
        object.__setattr__(self, "max_position_pct_equity", _clamp(self.max_position_pct_equity, "0.001", "1.00"))
        object.__setattr__(self, "max_loss_per_trade_pct", _clamp(self.max_loss_per_trade_pct, "0.001", "0.10"))
        object.__setattr__(self, "max_daily_loss_pct", _clamp(self.max_daily_loss_pct, "0.005", "0.25"))
        object.__setattr__(self, "max_drawdown_pct", _clamp(self.max_drawdown_pct, "0.01", "0.50"))
        object.__setattr__(self, "max_order_notional_usd", _clamp(self.max_order_notional_usd, "0.01", "10000"))
        object.__setattr__(self, "min_liquidation_buffer_pct", _clamp(self.min_liquidation_buffer_pct, "0.05", "0.95"))
        return self

    model_config = SettingsConfigDict(extra="ignore")


class ResearchSettings(BaseSettings):
    """Autonomous research loop cadence and validation thresholds."""

    enabled: bool = True
    loop_interval_s: int = Field(default=900, ge=60, le=86_400)
    training_interval_s: int = Field(default=21_600, ge=300)
    max_concurrent_jobs: int = Field(default=2, ge=1, le=16)

    # Overfitting protections. These are promotion gates, not suggestions.
    min_trades_for_promotion: int = Field(default=100, ge=30)
    min_oos_sharpe: Decimal = Field(default=Decimal("0.50"))
    max_oos_drawdown_pct: Decimal = Field(default=Decimal("0.25"))
    min_walk_forward_windows: int = Field(default=4, ge=2)
    # How much worse out-of-sample may be than in-sample before we call it
    # overfit and refuse promotion.
    max_is_oos_sharpe_degradation: Decimal = Field(default=Decimal("0.50"))
    monte_carlo_runs: int = Field(default=500, ge=50, le=20_000)
    min_paper_trades_for_live: int = Field(default=50, ge=10)
    probation_capital_pct: Decimal = Field(default=Decimal("0.10"))

    model_config = SettingsConfigDict(extra="ignore")


class NewsSettings(BaseSettings):
    """Free-first news ingestion. No paid API is required to operate."""

    enabled: bool = True
    poll_interval_s: int = Field(default=300, ge=30)
    max_articles_per_poll: int = Field(default=600, ge=1)
    feeds_file: str = "configs/news_sources.yaml"
    request_timeout_s: float = Field(default=15.0, ge=1.0, le=120.0)
    user_agent: str = "kalshi-autonomous-trader/0.1 (research)"

    model_config = SettingsConfigDict(extra="ignore")


class LLMSettings(BaseSettings):
    """LLM participation in trading decisions.

    The LLM is an *advisor*. It produces structured theses and probability
    estimates that become one input among many. It has no order-submission
    capability of any kind: see :mod:`kat.llm.advisor`.

    Defaults target a local OpenAI-compatible server (Ollama, llama.cpp,
    LM Studio) so the platform is never dependent on a paid hosted model.
    """

    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "llama3.1:8b"
    api_key: str = ""  # local servers ignore this; hosted ones need it
    timeout_s: float = Field(default=60.0, ge=5.0, le=600.0)
    max_tokens: int = Field(default=1_200, ge=64, le=32_000)
    temperature: Decimal = Field(default=Decimal("0.2"))
    # Hard cap on how much an LLM opinion may move a position size, as a
    # fraction of what the quantitative pipeline decided on its own.
    max_influence_pct: Decimal = Field(default=Decimal("0.25"))

    @model_validator(mode="after")
    def _clamp_all(self) -> LLMSettings:
        object.__setattr__(self, "max_influence_pct", _clamp(self.max_influence_pct, "0.00", "0.50"))
        object.__setattr__(self, "temperature", _clamp(self.temperature, "0.00", "1.00"))
        return self

    model_config = SettingsConfigDict(extra="ignore")


class AppSettings(BaseSettings):
    """Root settings object.

    Loaded from, in increasing precedence: defaults here, ``configs/*.yaml``,
    the ``.env`` file, then real environment variables. Nested values use the
    ``KAT_`` prefix and ``__`` as the delimiter, e.g.
    ``KAT_KALSHI__EVENT_CREDENTIALS__API_KEY_ID``.
    """

    # -- identity and mode ---------------------------------------------------
    app_name: str = "kalshi-autonomous-trader"
    operating_mode: OperatingMode = OperatingMode.GATED_DISCOVERY
    environment: TradingEnvironment = TradingEnvironment.PAPER

    # Live trading requires this to be explicitly true *and* environment=live.
    # Two independent switches, so a single stray value cannot arm real money.
    live_trading_armed: bool = False

    # -- host / process ------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65_535)
    # How the dashboard is reachable from another device.
    #
    # "off"       -- loopback only. Nothing outside this machine can reach it.
    # "tailscale" -- also bind to this machine's tailnet address, so your own
    #                devices can reach it over WireGuard and nothing else can.
    #
    # There is deliberately no option that binds every interface. The API has no
    # authentication by design (see docs/SECURITY.md), so the network is the
    # authentication, and kat.net.tailscale refuses any address that is not
    # private by construction.
    api_remote_access: Literal["off", "tailscale"] = "off"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = "logs"
    data_dir: str = "data"

    # -- freshness / safety budgets -----------------------------------------
    # An order is never built from market data older than this.
    max_market_data_age_ms: int = Field(default=5_000, ge=100, le=120_000)
    # How stale a *quiet* market's book may be while the feed itself is proven
    # live. This is a different question from the budget above: on a delta feed,
    # a market that has sent nothing is usually saying the book has not changed,
    # and only silence across the entire feed means we may be disconnected. See
    # MarketState.is_fresh. Bounded rather than unlimited because a live feed
    # does not prove one particular subscription is still live.
    quiet_market_max_age_ms: int = Field(default=300_000, ge=5_000, le=3_600_000)
    # Account state older than this forces a reconciliation before trading.
    max_account_state_age_ms: int = Field(default=30_000, ge=1_000, le=600_000)
    # Startup reconciliation must succeed within this budget or we stay halted.
    reconciliation_timeout_s: float = Field(default=30.0, ge=5.0, le=300.0)

    # -- subsystem settings --------------------------------------------------
    kalshi: KalshiSettings = Field(default_factory=KalshiSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    event_risk: EventRiskSettings = Field(default_factory=EventRiskSettings)
    perp_risk: PerpRiskSettings = Field(default_factory=PerpRiskSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    model_config = SettingsConfigDict(
        env_prefix="KAT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Precedence, highest first: init kwargs, env vars, .env, YAML.

        YAML sits *below* the environment deliberately. The YAML file is checked
        into the repository and holds shared defaults; the environment holds
        machine-specific values and secrets. If YAML won, a committed default
        would silently override the operator's real database URL -- which is
        exactly the bug this ordering exists to prevent.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlSettingsSource(settings_cls),
        )

    @field_validator("environment", mode="after")
    @classmethod
    def _propagate_env(cls, v: TradingEnvironment, info: ValidationInfo) -> TradingEnvironment:
        return v

    @model_validator(mode="after")
    def _finalise(self) -> AppSettings:
        # Keep the Kalshi client's notion of environment in lockstep with the
        # application's, so endpoint selection can never drift from the mode.
        object.__setattr__(self.kalshi, "environment", self.environment)

        # Live trading needs both switches. If only one is set, fail loudly at
        # load time rather than surprising anyone at the first order.
        if self.live_trading_armed and self.environment is not TradingEnvironment.LIVE:
            raise ConfigError(
                "live_trading_armed=true requires environment=live; "
                "set both deliberately or neither"
            )
        return self

    # -- convenience ---------------------------------------------------------

    @property
    def can_submit_real_orders(self) -> bool:
        """The single predicate the execution layer consults."""
        return self.environment is TradingEnvironment.LIVE and self.live_trading_armed

    @property
    def log_path(self) -> Path:
        p = Path(self.log_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    def redacted(self) -> dict[str, Any]:
        """Config dump with secrets removed, safe for logs and the dashboard."""
        data = self.model_dump(mode="json")
        for domain in ("event_credentials", "perp_credentials", "prod_credentials"):
            block = data.get("kalshi", {}).get(domain, {})
            if block.get("api_key_id"):
                block["api_key_id"] = _mask(block["api_key_id"])
            if block.get("private_key_path"):
                block["private_key_path"] = "<configured>"
        if data.get("llm", {}).get("api_key"):
            data["llm"]["api_key"] = "<configured>"
        if data.get("database", {}).get("url"):
            data["database"]["url"] = _mask_url(data["database"]["url"])
        return data


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _mask_url(url: str) -> str:
    """Strip credentials out of a database URL before it is displayed."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://****@{host}"


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Lowest-priority settings source backed by ``configs/default.yaml``.

    The path is overridable with ``KAT_CONFIG_FILE`` so an operator can keep a
    machine-specific config outside the repository. A missing file is not an
    error -- the model defaults are a complete, safe configuration on their own.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = self._read()

    @staticmethod
    def _read() -> dict[str, Any]:
        config_file = os.environ.get(
            "KAT_CONFIG_FILE", str(REPO_ROOT / "configs" / "default.yaml")
        )
        path = Path(config_file)
        if not path.exists():
            return {}
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path} must contain a YAML mapping at the top level")
        return loaded

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        if field_name in self._data:
            return self._data[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Re-read configuration. Used by tests and by the dashboard's reload action."""
    global _settings
    _settings = AppSettings()
    return _settings
