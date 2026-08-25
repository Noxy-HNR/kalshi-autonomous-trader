"""Native application entry point.

    python -m kat.engine.entrypoint

Starts one long-lived process containing the trading engine, market-data
ingestion, the research loop and the dashboard API. No container, no supervisor
script, no shell wrapper, and no subprocess for any ordinary trading operation.

Why the API lives here rather than in its own service
-----------------------------------------------------
The dashboard reads the engine's in-memory state. A process boundary would mean
either polling the database -- which is stale and puts the database back on a
path it was deliberately removed from -- or building an IPC layer for a single
consumer. Uvicorn is run programmatically on the same event loop instead.

What this will and will not do on start-up
------------------------------------------
It will connect, authenticate, discover capabilities, reconcile, and begin
ingesting. It will **not** trade with real money unless both ``environment=live``
and ``live_trading_armed=true`` are set, and it will not trade at all until the
whole start-up sequence has completed cleanly. A first run on a fresh machine
ends up watching markets in paper mode, which is the correct place to end up.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from kat.arbitrage.scanner import ArbitrageScanner, opportunity_to_intents
from kat.common.enums import MarketType, StrategyState, TradingEnvironment
from kat.common.errors import ConfigError
from kat.common.timeutil import SystemClock
from kat.config.settings import AppSettings, get_settings
from kat.engine.context import EngineContext, set_context
from kat.engine.decision import DecisionRecorder
from kat.engine.features import FeatureWorker
from kat.engine.ingestion import (
    MarketDataIngestor,
    select_event_watchlist,
    select_perp_watchlist,
)
from kat.engine.lifecycle import Application, Phase, ServiceComponent
from kat.engine.trading import TradingLoop
from kat.execution.event_exec import EventExecutionEngine
from kat.execution.perp_exec import PerpExecutionEngine
from kat.execution.reconcile import Reconciler
from kat.kalshi.auth import KalshiSigner
from kat.kalshi.event_contracts.rest import EventContractsREST
from kat.kalshi.http import KalshiHTTPClient
from kat.kalshi.perps.rest import PerpsREST
from kat.kalshi.ratelimit import RateLimiter
from kat.kalshi.ws import KalshiWebSocket
from kat.marketdata.state import MarketState
from kat.models.inference import InferenceEngine
from kat.models.registry import ModelRegistry
from kat.news.pipeline import NewsPipeline
from kat.observability.logging import configure_logging, get_logger
from kat.research.loop import ResearchLoop
from kat.risk.event_risk import EventAccountState, EventRiskEngine
from kat.risk.invariants import MAX_PERP_LEVERAGE, assert_invariant_intact
from kat.risk.perp_risk import PerpAccountState, PerpRiskEngine
from kat.strategies.registry import StrategyRegistry, build_default_strategies

log = get_logger(__name__)

__all__ = ["build_application", "main", "run"]


class Platform:
    """Assembles every subsystem and hands the application its components."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.clock = SystemClock()
        self.app = Application(settings)
        self.context = EngineContext(app=self.app)

        self.rate_limiter = RateLimiter(
            event_read_per_s=settings.kalshi.read_tokens_per_second,
            event_write_per_s=settings.kalshi.write_tokens_per_second,
            perp_read_per_s=settings.kalshi.perp_read_tokens_per_second,
            perp_write_per_s=settings.kalshi.perp_write_tokens_per_second,
            utilisation=float(settings.kalshi.rate_limit_utilisation),
        )

        self.event_http: KalshiHTTPClient | None = None
        self.perp_http: KalshiHTTPClient | None = None
        self.event_rest: EventContractsREST | None = None
        self.perp_rest: PerpsREST | None = None
        self.perps_enabled = False

        # Populated during build. Declared here so an attribute that has not
        # been built yet reads as absent rather than raising AttributeError from
        # a component body that started early.
        self.feature_workers: dict[MarketType, FeatureWorker] = {}
        self.inference_engines: dict[MarketType, InferenceEngine] = {}
        self.news: NewsPipeline | None = None
        self.arbitrage: ArbitrageScanner | None = None
        self.research: ResearchLoop | None = None

    # -- construction --------------------------------------------------------

    def build_clients(self) -> None:
        """Create REST clients for whichever domains have credentials.

        A missing credential is not an error: the platform runs perfectly well
        watching public market data with no keys at all, which is exactly what a
        first run should do.
        """
        kalshi = self.settings.kalshi

        event_signer = None
        if kalshi.event_credentials.configured:
            try:
                event_signer = KalshiSigner.from_credentials(
                    kalshi.event_credentials, label="event"
                )
                log.info("kalshi.event_credentials_loaded", signer=repr(event_signer))
            except ConfigError as exc:
                log.error("kalshi.event_credentials_invalid", error=str(exc))
        else:
            log.warning(
                "kalshi.event_credentials_absent",
                note="public market data only; no account or order access",
            )

        perp_signer = None
        if kalshi.perp_credentials.configured:
            try:
                perp_signer = KalshiSigner.from_credentials(
                    kalshi.perp_credentials, label="perp"
                )
                log.info("kalshi.perp_credentials_loaded", signer=repr(perp_signer))
            except ConfigError as exc:
                log.error("kalshi.perp_credentials_invalid", error=str(exc))

        self.event_http = KalshiHTTPClient(
            base_url=kalshi.rest_base_url,
            signer=event_signer,
            rate_limiter=self.rate_limiter,
            is_perp=False,
            timeout_s=kalshi.request_timeout_s,
            connect_timeout_s=kalshi.connect_timeout_s,
        )
        self.event_rest = EventContractsREST(self.event_http)

        self.perp_http = KalshiHTTPClient(
            base_url=kalshi.perp_rest_base_url,
            signer=perp_signer,
            rate_limiter=self.rate_limiter,
            is_perp=True,
            timeout_s=kalshi.request_timeout_s,
            connect_timeout_s=kalshi.connect_timeout_s,
        )
        self.perp_rest = PerpsREST(self.perp_http)

    async def discover_capabilities(self) -> None:
        """Ask the exchange what this account can actually do.

        Perpetuals roll out member by member, so the perp subsystem stays
        dormant rather than failing every request when it is not enabled.
        """
        assert self.event_rest is not None
        try:
            status = await self.event_rest.get_exchange_status()
            self.app.report.kalshi_event_ready = bool(status.get("exchange_active"))
            log.info("kalshi.exchange_status", **{k: v for k, v in status.items() if k != "exchange_index_statuses"})
        except Exception as exc:
            log.error("kalshi.exchange_status_failed", error=str(exc))
            # Latched: the next health probe cannot tell whether the exchange
            # came back, and re-permitting trading against an exchange we never
            # reached is precisely what this block exists to prevent. Cleared by
            # the reconciliation cycle, which does contact the exchange.
            self.app.add_block("kalshi", f"Kalshi is unreachable: {exc}")

        # Real tier budgets, so the limiter stops guessing.
        if self.settings.kalshi.event_credentials.configured:
            try:
                limits = await self.event_rest.get_account_limits()
                read, write = limits.get("read") or {}, limits.get("write") or {}
                if read:
                    self.rate_limiter.apply_tier_limits(
                        "event_read", read.get("refill_rate", 200), read.get("bucket_capacity", 200)
                    )
                if write:
                    self.rate_limiter.apply_tier_limits(
                        "event_write", write.get("refill_rate", 100),
                        write.get("bucket_capacity", 200),
                    )
                costs = await self.event_rest.get_endpoint_costs()
                if costs:
                    self.rate_limiter.set_endpoint_costs(costs)
            except Exception as exc:
                log.warning("kalshi.limits_unavailable", error=str(exc))

        if self.settings.kalshi.perp_credentials.configured and self.perp_rest is not None:
            try:
                self.perps_enabled = await self.perp_rest.is_margin_enabled()
                self.app.report.perps_enabled_for_account = self.perps_enabled
                self.app.report.kalshi_perp_ready = self.perps_enabled
                log.info("kalshi.perps_enabled", enabled=self.perps_enabled)
                if not self.perps_enabled:
                    log.warning(
                        "kalshi.perps_dormant",
                        note="margin is not enabled for this account in this environment",
                    )
            except Exception as exc:
                self.perps_enabled = False
                self.app.report.perps_enabled_for_account = False
                log.warning("kalshi.perps_check_failed", error=str(exc))

    def build_trading(self) -> None:
        """Create market state, risk, execution and a trading loop per domain."""
        settings = self.settings
        writer = self.app.writer
        environment = settings.environment

        event_state = MarketState(MarketType.EVENT, max_age_ms=settings.max_market_data_age_ms)
        perp_state = MarketState(MarketType.PERP, max_age_ms=settings.max_market_data_age_ms)
        self.context.market_states = {
            MarketType.EVENT: event_state,
            MarketType.PERP: perp_state,
        }

        event_exec = EventExecutionEngine(
            rest=self.event_rest,
            market_state=event_state,
            clock=self.clock,
            writer=writer,
            kill_switches=self.app.kill_switches,
            environment=environment,
            live_armed=settings.live_trading_armed,
        )
        perp_exec = PerpExecutionEngine(
            rest=self.perp_rest,
            market_state=perp_state,
            clock=self.clock,
            writer=writer,
            kill_switches=self.app.kill_switches,
            environment=environment,
            live_armed=settings.live_trading_armed,
        )

        recorder_kwargs = {"environment": environment, "app_version": "0.1.0"}
        event_loop = TradingLoop(
            market_type=MarketType.EVENT,
            market_state=event_state,
            risk_engine=EventRiskEngine(
                settings.event_risk, max_state_age_ms=settings.max_account_state_age_ms
            ),
            execution=event_exec,
            recorder=DecisionRecorder(writer, **recorder_kwargs),
            clock=self.clock,
            kill_switches=self.app.kill_switches,
        )
        perp_loop = TradingLoop(
            market_type=MarketType.PERP,
            market_state=perp_state,
            risk_engine=PerpRiskEngine(
                settings.perp_risk, max_state_age_ms=settings.max_account_state_age_ms
            ),
            execution=perp_exec,
            recorder=DecisionRecorder(writer, **recorder_kwargs),
            clock=self.clock,
            kill_switches=self.app.kill_switches,
        )

        # Paper mode has no exchange account, so seed a starting balance from
        # config rather than leaving the loop with no state to size against.
        if environment is TradingEnvironment.PAPER:
            now_ms = self.clock.now_ms()
            event_loop.account_state = EventAccountState(
                balance_micros=200_000_000, updated_at_ms=now_ms
            )
            event_loop.account_state.peak_equity_micros = event_loop.account_state.equity_micros
            perp_loop.account_state = PerpAccountState(
                equity_micros=200_000_000,
                equity_source="paper_seed",
                equity_is_known=True,
                updated_at_ms=now_ms,
            )

        # Register the shipped strategies at PAPER: they trade with simulated
        # fills against live market data, which is the lowest rung that produces
        # evidence. Nothing here has earned live capital, and nothing reaches it
        # without climbing the ladder.
        registry = StrategyRegistry(writer)
        for strategy in build_default_strategies(state=StrategyState.PAPER):
            registry.register(strategy)
            loop = event_loop if strategy.market_type is MarketType.EVENT else perp_loop
            loop.add_strategy(strategy)
        self.context.strategy_registry = registry
        log.info(
            "strategies.loaded",
            total=len(registry),
            # Not `event=`: structlog's own first parameter is named `event`, so
            # passing it as a keyword collides with the message itself.
            event_contracts=len(registry.all(market_type=MarketType.EVENT)),
            perps=len(registry.all(market_type=MarketType.PERP)),
            state=StrategyState.PAPER.value,
        )

        # Features come before everything else that thinks. Without them every
        # strategy skips on `*_unavailable`, the risk engine is never consulted,
        # and the platform looks healthy while doing nothing at all.
        self.feature_workers = {
            MarketType.EVENT: FeatureWorker(
                MarketType.EVENT, self.context.market_states[MarketType.EVENT], self.clock
            ),
            MarketType.PERP: FeatureWorker(
                MarketType.PERP, self.context.market_states[MarketType.PERP], self.clock
            ),
        }

        # Inference serves whatever the registry has deployed, or nothing. A
        # model has no order authority: its output reaches a strategy as one
        # input and reaches the audit record as provenance.
        model_registry = ModelRegistry(self.app.database) if self.app.database else None
        self.inference_engines = (
            {
                market_type: InferenceEngine(model_registry, market_type)
                for market_type in (MarketType.EVENT, MarketType.PERP)
            }
            if model_registry is not None
            else {}
        )

        # Arbitrage is cross-market, so it enters through an intent source
        # rather than the per-ticker strategy sweep. Its intents then travel the
        # identical path: risk engine, decision record, execution.
        self.arbitrage = ArbitrageScanner(
            market_state=self.context.market_states[MarketType.EVENT],
            clock=self.clock,
            writer=self.app.writer,
        )
        event_loop.add_intent_source("arbitrage", self._scan_arbitrage)

        self.context.loops = {MarketType.EVENT: event_loop, MarketType.PERP: perp_loop}
        self.research = ResearchLoop(
            settings=settings.research,
            registry=registry,
            market_states=self.context.market_states,
            loops=self.context.loops,
            clock=self.clock,
            mode=settings.operating_mode,
        )
        self.context.research_loop = self.research

        self.context.reconciler = Reconciler(
            event_rest=self.event_rest,
            perp_rest=self.perp_rest,
            event_exec=event_exec,
            perp_exec=perp_exec,
            kill_switches=self.app.kill_switches,
        )

    def register_components(self) -> None:
        """Register everything the application supervises, in start order."""
        settings = self.settings

        event_ws = KalshiWebSocket(
            url=settings.kalshi.ws_url,
            signer=self.event_http.signer if self.event_http else None,
            domain="event",
            ping_interval_s=settings.kalshi.ws_ping_interval_s,
            ping_timeout_s=settings.kalshi.ws_ping_timeout_s,
            max_reconnect_delay_s=settings.kalshi.ws_max_reconnect_delay_s,
        )
        event_ingestor = MarketDataIngestor(
            market_type=MarketType.EVENT,
            ws=event_ws,
            state=self.context.market_states[MarketType.EVENT],
            clock=self.clock,
            writer=self.app.writer,
        )

        async def start_event_ingestion() -> None:
            # Market data requires an authenticated socket even for public
            # channels, so without credentials this stays down by design.
            if event_ws.signer is None:
                log.warning(
                    "ingestion.event_disabled",
                    note="Kalshi requires an authenticated WebSocket even for public channels",
                )
                return
            await event_ingestor.start()
            if self.event_rest is not None:
                watchlist = await select_event_watchlist(self.event_rest)
                await event_ingestor.set_watchlist(watchlist)

        self.app.register(
            ServiceComponent(
                "event-market-data",
                start=start_event_ingestion,
                stop=event_ingestor.stop,
                health=lambda: event_ws.signer is None or event_ingestor.healthy(),
                critical=False,
            )
        )

        if self.perps_enabled:
            perp_ws = KalshiWebSocket(
                url=settings.kalshi.perp_ws_url,
                signer=self.perp_http.signer if self.perp_http else None,
                domain="perp",
                ping_interval_s=settings.kalshi.ws_ping_interval_s,
                ping_timeout_s=settings.kalshi.ws_ping_timeout_s,
                max_reconnect_delay_s=settings.kalshi.ws_max_reconnect_delay_s,
            )
            perp_ingestor = MarketDataIngestor(
                market_type=MarketType.PERP,
                ws=perp_ws,
                state=self.context.market_states[MarketType.PERP],
                clock=self.clock,
                writer=self.app.writer,
            )

            async def start_perp_ingestion() -> None:
                await perp_ingestor.start()
                if self.perp_rest is not None:
                    await perp_ingestor.set_watchlist(await select_perp_watchlist(self.perp_rest))

            self.app.register(
                ServiceComponent(
                    "perp-market-data",
                    start=start_perp_ingestion,
                    stop=perp_ingestor.stop,
                    health=perp_ingestor.healthy,
                    critical=False,
                )
            )

        # Reconciliation runs once at start-up and then on a slow cycle.
        self.app.register(
            ServiceComponent("reconciliation", start=self._reconcile_once, run=self._reconcile_loop)
        )

        for market_type, loop in self.context.loops.items():
            if market_type is MarketType.PERP and not self.perps_enabled:
                continue

            worker = self.feature_workers[market_type]
            self.app.register(
                ServiceComponent(
                    f"{market_type.value}-features",
                    run=_bind(worker.run, loop),
                    health=worker.healthy,
                    critical=False,
                )
            )

            inference = self.inference_engines.get(market_type)
            if inference is not None:
                self.app.register(
                    ServiceComponent(
                        f"{market_type.value}-inference",
                        run=_bind(inference.run, loop),
                        critical=False,
                    )
                )

            self.app.register(
                ServiceComponent(
                    f"{market_type.value}-trading",
                    run=loop.run,
                    stop=_async_noop(loop.stop),
                    critical=False,
                )
            )

        if settings.research.enabled:
            self.app.register(
                ServiceComponent(
                    "research-regimes",
                    run=_bind(self.research.run_regimes),
                    critical=False,
                )
            )
            self.app.register(
                ServiceComponent("research", run=self.research.run, critical=False)
            )

        if settings.news.enabled:
            self.news = NewsPipeline(settings.news, self.clock, writer=self.app.writer)
            self.app.register(
                ServiceComponent(
                    "news",
                    start=_async_call(self.news.load_sources),
                    run=self.news.run,
                    stop=self.news.aclose,
                    health=self.news.healthy,
                    critical=False,
                )
            )
            self.app.register(
                ServiceComponent("news-fanout", run=self._news_fanout, critical=False)
            )

        self.app.register(
            ServiceComponent("dashboard-api", start=self._start_api, stop=self._stop_api)
        )

    # -- component bodies ----------------------------------------------------

    def _scan_arbitrage(self, now_ms: int) -> list[Any]:
        """One arbitrage pass, converted to intents.

        Returns an empty list on any failure. A scanner that raises must not
        stop the trading cycle: the strategies still have opinions worth acting
        on, and an arbitrage that was not found is not an emergency.
        """
        try:
            opportunities = self.arbitrage.scan()
        except Exception as exc:
            log.error("arbitrage.scan_failed", error=str(exc), exc_info=True)
            return []

        intents: list[Any] = []
        for opportunity in opportunities:
            intents.extend(opportunity_to_intents(opportunity))
        if intents:
            log.info(
                "arbitrage.intents",
                opportunities=len(opportunities), intents=len(intents), now_ms=now_ms,
            )
        return intents

    async def _news_fanout(self) -> None:
        """Publish news signals into each loop, and market profiles into the matcher.

        Kept separate from the pipeline's own poll loop because the two run at
        different cadences for different reasons: fetching is slow and polite to
        the publishers, while the loops need whatever is already known refreshed
        against the current watchlist.
        """
        from kat.news.matching import MarketProfile

        while True:
            try:
                if self.news is None:  # pragma: no cover - not registered without news
                    return
                now_ms = self.clock.now_ms()
                profiles = []
                for market_type, state in self.context.market_states.items():
                    if market_type is MarketType.PERP and not self.perps_enabled:
                        continue
                    for ticker in state.tickers():
                        snapshot = state.get(ticker)
                        if snapshot is None:
                            continue
                        meta = snapshot.metadata
                        profiles.append(
                            MarketProfile.build(
                                ticker=ticker,
                                market_type=market_type,
                                title=str(meta.get("title", "")),
                                event_ticker=str(meta.get("event_ticker", "")),
                                series_ticker=str(meta.get("series_ticker", "")),
                                rules=str(meta.get("rules_primary", "")),
                                close_ms=(
                                    meta.get("close_ts_ms")
                                    if isinstance(meta.get("close_ts_ms"), int)
                                    else None
                                ),
                            )
                        )
                self.news.set_markets(profiles)

                # signals_at filters on ingestion time, so a loop only ever sees
                # what this process could actually have acted on.
                signals = self.news.signals_at(now_ms)
                for market_type, loop in self.context.loops.items():
                    if market_type is MarketType.PERP and not self.perps_enabled:
                        continue
                    loop.research = {
                        ticker: blob
                        for ticker, blob in signals.items()
                        if self.context.market_states[market_type].get(ticker) is not None
                    }
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("news.fanout_failed", error=str(exc), exc_info=True)
            await asyncio.sleep(30)


    async def _reconcile_once(self) -> None:
        """Reconcile before any trading is permitted.

        In paper mode there is no exchange account to reconcile against, so the
        seeded state stands and the step is skipped explicitly rather than
        silently passing.
        """
        if self.settings.environment is TradingEnvironment.PAPER:
            self.app.report.reconciled = True
            # No exchange account exists to reconcile against, so a start-up
            # probe failure must not latch trading off forever in paper mode.
            self.app.clear_block("reconcile:event")
            self.app.clear_block("reconcile:perp")
            self.app.clear_block("kalshi")
            log.info("reconcile.skipped_paper_mode")
            self._publish_trading_gate()
            return
        await self._reconcile()

    async def _reconcile(self) -> None:
        reconciler = self.context.reconciler
        if reconciler is None:
            return
        now_ms = self.clock.now_ms()
        event_loop = self.context.loops[MarketType.EVENT]
        if event_loop.account_state is None:
            event_loop.account_state = EventAccountState()
        report = await reconciler.reconcile_event_contracts(event_loop.account_state, now_ms=now_ms)
        if await reconciler.halt_if_unsafe(report):
            # "We do not know what we hold" is not a condition a liveness probe
            # can disprove, so it stays latched until a reconciliation succeeds.
            self.app.add_block("reconcile:event", "event reconciliation failed")
        else:
            self.app.clear_block("reconcile:event")
            self.app.clear_block("kalshi")
            if report.orphaned_orders:
                await reconciler.cancel_orphans(report, MarketType.EVENT)

        if self.perps_enabled:
            perp_loop = self.context.loops[MarketType.PERP]
            if perp_loop.account_state is None:
                perp_loop.account_state = PerpAccountState()
            perp_report = await reconciler.reconcile_perpetuals(
                perp_loop.account_state, now_ms=now_ms
            )
            if await reconciler.halt_if_unsafe(perp_report):
                self.app.add_block(
                    "reconcile:perp", "perpetual reconciliation failed"
                )
            else:
                self.app.clear_block("reconcile:perp")
                if perp_report.orphaned_orders:
                    await reconciler.cancel_orphans(perp_report, MarketType.PERP)

        self.app.report.reconciled = True
        self._publish_trading_gate()

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(60.0)
            try:
                if self.settings.environment is not TradingEnvironment.PAPER:
                    await self._reconcile()
                else:
                    # Keep the seeded paper state fresh so the risk engines do
                    # not reject on staleness.
                    now_ms = self.clock.now_ms()
                    for loop in self.context.loops.values():
                        if loop.account_state is not None:
                            loop.account_state.updated_at_ms = now_ms
                self._publish_trading_gate()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("reconcile.cycle_failed", error=str(exc), exc_info=True)

    def _publish_trading_gate(self) -> None:
        """Propagate the application's trading gate down to each loop."""
        permitted = self.app.phase is Phase.RUNNING and self.app.trading_permitted
        for loop in self.context.loops.values():
            loop.trading_permitted = permitted
            loop.operating_mode = self.settings.operating_mode

    async def _start_api(self) -> None:
        import uvicorn
        from apps.api.main import app as fastapi_app

        config = uvicorn.Config(
            fastapi_app,
            host=self.settings.api_host,
            port=self.settings.api_port,
            log_level="warning",
            access_log=False,
        )
        self._api_server = uvicorn.Server(config)
        # Uvicorn installs its own signal handlers by default, which would
        # compete with the application's own shutdown path.
        self._api_server.install_signal_handlers = False
        self._api_task = asyncio.create_task(self._api_server.serve(), name="dashboard-api")
        log.info(
            "api.started",
            url=f"http://{self.settings.api_host}:{self.settings.api_port}",
        )

    async def _stop_api(self) -> None:
        server = getattr(self, "_api_server", None)
        task = getattr(self, "_api_task", None)
        if server is not None:
            server.should_exit = True
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()

    async def aclose(self) -> None:
        for client in (self.event_http, self.perp_http):
            if client is not None:
                await client.aclose()


def _async_noop(fn: Any):
    async def _stop() -> None:
        fn()

    return _stop


def _bind(fn: Any, *args: Any):
    """A no-argument coroutine factory, for a component body that needs one.

    ServiceComponent bodies take no arguments; the feature and inference workers
    need the loop they publish into. A lambda would capture the loop variable by
    reference and, inside a for-loop, every component would end up bound to the
    last iteration's loop.
    """

    async def _run() -> None:
        await fn(*args)

    return _run


def _async_call(fn: Any):
    """Adapt a synchronous start-up step to the async component contract."""

    async def _start() -> None:
        fn()

    return _start


async def build_application(settings: AppSettings | None = None) -> Platform:
    """Assemble the platform without starting it. Used by tests."""
    settings = settings or get_settings()
    platform = Platform(settings)
    platform.build_clients()
    set_context(platform.context)
    return platform


async def run(settings: AppSettings | None = None) -> int:
    """Start everything and block until shutdown. Returns a process exit code."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, log_dir=settings.log_path)

    # Verify the leverage invariant before anything else exists. A compromised
    # build must fail at start-up, not at the first order.
    assert_invariant_intact()
    log.info(
        "invariant.verified",
        max_perp_leverage=f"{MAX_PERP_LEVERAGE:.2f}",
        note="behavioural canaries passed",
    )

    platform = await build_application(settings)
    try:
        await platform.discover_capabilities()
        platform.build_trading()

        # The database must be up before components that write to it start.
        await platform.app._start_database()
        await platform.app._load_persisted_state()

        platform.register_components()
        report = await platform.app.start()
        platform._publish_trading_gate()

        banner(settings, report.trading_permitted)
        await platform.app.run_until_stopped()
        return 0
    except KeyboardInterrupt:
        log.info("app.interrupted")
        return 0
    except Exception as exc:
        log.error("app.fatal", error=str(exc), exc_info=True)
        return 1
    finally:
        await platform.app.stop(clean=True)
        await platform.aclose()


def banner(settings: AppSettings, trading_permitted: bool) -> None:
    """Say plainly what mode this process is in. Ambiguity here is dangerous."""
    environment = settings.environment.value.upper()
    real_money = settings.can_submit_real_orders
    lines = [
        "",
        "=" * 68,
        "  KALSHI AUTONOMOUS TRADER",
        f"  ENVIRONMENT: {environment}" + ("   *** REAL MONEY ***" if real_money else ""),
        f"  MODE:        {settings.operating_mode.value}",
        f"  TRADING:     {'permitted' if trading_permitted else 'HALTED'}",
        f"  MAX PERP LEVERAGE: {MAX_PERP_LEVERAGE:.2f}x  (architectural, not configurable)",
        f"  DASHBOARD:   http://{settings.api_host}:{settings.api_port}",
        "=" * 68,
        "",
    ]
    # Printed rather than logged: this is the operator's confirmation of what
    # the process is about to do, and it must be visible even at WARNING level.
    print("\n".join(lines))


def main() -> None:
    """Console entry point (``kat-trader``)."""
    try:
        sys.exit(asyncio.run(run()))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
