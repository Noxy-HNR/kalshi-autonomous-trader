"""Dashboard API.

Runs in the trading engine's own process (see :mod:`kat.engine.context` for why),
serving live in-memory state rather than polling the database for things the
engine already knows.

Read endpoints are plain reads. Write endpoints are the enumerated control
actions and nothing else -- there is no endpoint that submits an order, no
endpoint that raises a risk limit past its backend clamp, and no endpoint that
touches perpetual leverage, because no such field exists to touch.

Binds to 127.0.0.1 by default. It exposes kill switches and mode changes, so
exposing it on a public interface would be a mistake; see docs/SECURITY.md.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select

from kat.common.enums import KillSwitchScope, MarketType, OperatingMode
from kat.common.fixedpoint import micros_to_decimal, price_cc_to_decimal, qc_to_decimal
from kat.db.engine import get_database
from kat.db.models import (
    ArbitrageOpportunity,
    AuditLog,
    Decision,
    Fill,
    LLMSignal,
    Market,
    ModelRegistry,
    NewsArticle,
    NewsMarketLink,
    Order,
    RiskEvent,
    Strategy,
    StrategyStateChange,
)
from kat.docs import build_docs
from kat.engine.context import get_context
from kat.observability.logging import get_logger
from kat.risk.invariants import MAX_PERP_LEVERAGE
from kat.risk.tunables import TunableError, current_values

log = get_logger(__name__)

app = FastAPI(
    title="Kalshi Autonomous Trader",
    version="0.1.0",
    description=(
        "Control and monitoring API. Perpetual leverage is capped at 1.00x by an "
        "architectural invariant; no endpoint here can change it."
    ),
)

# The dashboard dev server runs on a different port. Locked to localhost origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")


# =============================================================================
# Overview and system
# =============================================================================


@api.get("/overview")
async def overview() -> dict[str, Any]:
    """Equity, P&L, exposure, active strategies, health -- the landing tab."""
    return get_context().overview()


@api.get("/health")
async def health() -> dict[str, Any]:
    return get_context().health()


@api.get("/system")
async def system() -> dict[str, Any]:
    """Process and host resources, queue depths, latency, data freshness."""
    return get_context().system()


@api.get("/snapshot")
async def snapshot() -> dict[str, Any]:
    """Everything the dashboard needs in one poll."""
    return get_context().snapshot()


# =============================================================================
# Perpetuals -- the leverage surface
# =============================================================================


@api.get("/perps/leverage")
async def perp_leverage() -> dict[str, Any]:
    """Current and projected leverage, the hard maximum, and blocked violations."""
    return get_context().leverage_panel()


@api.get("/perps/violations")
async def perp_leverage_violations(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """Every leverage-related rejection, with the arithmetic behind each one."""
    db = get_database()
    async with db.session() as session:
        rows = (
            await session.execute(
                select(RiskEvent)
                .where(RiskEvent.reason_code.like("LEVERAGE%"))
                .order_by(desc(RiskEvent.ts))
                .limit(limit)
            )
        ).scalars().all()
    return {
        "max_leverage": f"{MAX_PERP_LEVERAGE:.2f}",
        "violations": [
            {
                "ts": r.ts.isoformat() if r.ts else None,
                "ticker": r.ticker,
                "strategy_id": r.strategy_id,
                "reason_code": r.reason_code,
                "message": r.message,
                "severity": r.severity,
                "leverage": (r.detail or {}).get("leverage"),
                "decision_id": r.decision_id,
            }
            for r in rows
        ],
    }


@api.get("/perps/positions")
async def perp_positions() -> dict[str, Any]:
    ctx = get_context()
    loop = ctx.loops.get(MarketType.PERP)
    if loop is None or loop.account_state is None:
        return {"available": False, "positions": []}
    state = loop.account_state
    return {
        "available": True,
        "max_leverage": f"{MAX_PERP_LEVERAGE:.2f}",
        "positions": [
            {
                "ticker": p.ticker,
                "position": str(qc_to_decimal(p.position_qc)),
                "entry_price": (
                    str(price_cc_to_decimal(p.entry_price_cc)) if p.entry_price_cc else None
                ),
                "mark_price": (
                    str(price_cc_to_decimal(p.mark_price_cc)) if p.mark_price_cc else None
                ),
                "notional_usd": str(micros_to_decimal(p.exposure_micros())),
                "is_long": p.position_qc > 0,
            }
            for p in state.positions.values()
            if p.position_qc != 0
        ],
    }


# =============================================================================
# Event contracts
# =============================================================================


@api.get("/events/positions")
async def event_positions() -> dict[str, Any]:
    ctx = get_context()
    loop = ctx.loops.get(MarketType.EVENT)
    if loop is None or loop.account_state is None:
        return {"available": False, "positions": []}
    state = loop.account_state
    return {
        "available": True,
        "positions": [
            {
                "ticker": p.ticker,
                "event_ticker": p.event_ticker,
                "position": str(qc_to_decimal(p.position_qc)),
                "side": "yes" if p.position_qc > 0 else "no",
                "avg_entry": str(price_cc_to_decimal(p.avg_entry_cc)) if p.avg_entry_cc else None,
                "exposure_usd": str(micros_to_decimal(p.exposure_micros)),
                "unrealized_pnl_usd": str(micros_to_decimal(p.unrealized_pnl_micros)),
                "strategy_id": None,
            }
            for p in state.positions.values()
            if p.position_qc != 0
        ],
    }


@api.get("/markets")
async def markets(
    market_type: Literal["event", "perp"] | None = None,
    limit: int = Query(200, ge=1, le=2000),
) -> dict[str, Any]:
    """Live market inventory with spreads, liquidity and freshness."""
    ctx = get_context()
    now_ms = ctx.app.clock.now_ms() if ctx.app else 0
    out: list[dict[str, Any]] = []
    for domain, state in ctx.market_states.items():
        if market_type and domain.value != market_type:
            continue
        for ticker in state.tickers()[:limit]:
            snap = state.get(ticker)
            if snap is None:
                continue
            book = snap.book
            out.append(
                {
                    "ticker": ticker,
                    "market_type": domain.value,
                    "event_ticker": snap.event_ticker,
                    "status": snap.status,
                    "bid": str(price_cc_to_decimal(book.best_bid_cc)) if book and book.best_bid_cc else None,
                    "ask": str(price_cc_to_decimal(book.best_ask_cc)) if book and book.best_ask_cc else None,
                    "spread_cc": book.spread_cc if book else None,
                    "mid": str(price_cc_to_decimal(book.mid_cc)) if book and book.mid_cc else None,
                    "depth_bid_qc": book.total_depth_qc(is_bid=True) if book else 0,
                    "depth_ask_qc": book.total_depth_qc(is_bid=False) if book else 0,
                    "mark": str(price_cc_to_decimal(snap.mark_price_cc)) if snap.mark_price_cc else None,
                    "volume_qc": snap.volume_qc,
                    "open_interest_qc": snap.open_interest_qc,
                    "age_ms": state.age_ms(ticker, now_ms=now_ms),
                    "fresh": state.is_fresh(ticker, now_ms=now_ms),
                }
            )
    return {"markets": out, "count": len(out)}


@api.get("/markets/{ticker}/orderbook")
async def orderbook(ticker: str, levels: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    ctx = get_context()
    for state in ctx.market_states.values():
        book = state.book(ticker)
        if book is not None:
            return book.snapshot(levels=levels)
    raise HTTPException(status_code=404, detail=f"no live book for {ticker}")


# =============================================================================
# Orders, decisions, audit
# =============================================================================


@api.get("/orders")
async def orders(
    state: str | None = None,
    market_type: Literal["event", "perp"] | None = None,
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    db = get_database()
    query = select(Order).order_by(desc(Order.created_at)).limit(limit)
    if state:
        query = query.where(Order.state == state)
    if market_type:
        query = query.where(Order.market_type == market_type)
    async with db.session() as session:
        rows = (await session.execute(query)).scalars().all()
    return {
        "orders": [
            {
                "order_id": o.order_id,
                "client_order_id": o.client_order_id,
                "exchange_order_id": o.exchange_order_id,
                "market_type": o.market_type,
                "ticker": o.ticker,
                "strategy_id": o.strategy_id,
                "state": o.state,
                "side": o.outcome_side or o.book_side,
                "limit_price": (
                    str(price_cc_to_decimal(o.limit_price_cc)) if o.limit_price_cc else None
                ),
                "requested": str(qc_to_decimal(o.requested_qty_qc)),
                "filled": str(qc_to_decimal(o.filled_qty_qc)),
                "avg_fill_price": (
                    str(price_cc_to_decimal(o.avg_fill_price_cc)) if o.avg_fill_price_cc else None
                ),
                "fees_usd": str(micros_to_decimal(o.fees_micros)),
                "slippage_cc": o.slippage_cc,
                "submit_latency_ms": o.submit_latency_ms,
                "reject_reason": o.reject_reason,
                "environment": o.environment,
                "submitted_ts": o.submitted_ts.isoformat() if o.submitted_ts else None,
            }
            for o in rows
        ]
    }


@api.get("/fills")
async def fills(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    db = get_database()
    async with db.session() as session:
        rows = (
            await session.execute(select(Fill).order_by(desc(Fill.ts)).limit(limit))
        ).scalars().all()
    return {
        "fills": [
            {
                "fill_id": f.fill_id,
                "order_id": f.order_id,
                "ticker": f.ticker,
                "market_type": f.market_type,
                "strategy_id": f.strategy_id,
                "ts": f.ts.isoformat() if f.ts else None,
                "price": str(price_cc_to_decimal(f.price_cc)),
                "count": str(qc_to_decimal(f.count_qc)),
                "fee_usd": str(micros_to_decimal(f.fee_micros)),
                "is_taker": f.is_taker,
            }
            for f in rows
        ]
    }


@api.get("/decisions")
async def decisions(
    decision: Literal["approved", "rejected", "resized"] | None = None,
    market_type: Literal["event", "perp"] | None = None,
    reason_code: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    """Every considered trade, including rejections. The explainability surface."""
    db = get_database()
    query = select(Decision).order_by(desc(Decision.ts)).limit(limit)
    if decision:
        query = query.where(Decision.decision == decision)
    if market_type:
        query = query.where(Decision.market_type == market_type)
    if reason_code:
        query = query.where(Decision.reason_code == reason_code)
    async with db.session() as session:
        rows = (await session.execute(query)).scalars().all()
    return {"decisions": [_decision_dict(d) for d in rows]}


@api.get("/decisions/{decision_id}")
async def decision_detail(decision_id: str) -> dict[str, Any]:
    """The full record for one decision: every input, check and outcome."""
    db = get_database()
    async with db.session() as session:
        row = (
            await session.execute(select(Decision).where(Decision.decision_id == decision_id))
        ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id}")
    detail = _decision_dict(row)
    detail.update(
        risk_checks=row.risk_checks,
        features=row.features,
        leverage_assessment=row.leverage_assessment,
        execution_result=row.execution_result,
        app_version=row.app_version,
        git_commit=row.git_commit,
    )
    return detail


def _decision_dict(d: Decision) -> dict[str, Any]:
    return {
        "decision_id": d.decision_id,
        "correlation_id": d.correlation_id,
        "ts": d.ts.isoformat() if d.ts else None,
        "market_type": d.market_type,
        "ticker": d.ticker,
        "strategy_id": d.strategy_id,
        "strategy_version": d.strategy_version,
        "model_id": d.model_id,
        "model_version": d.model_version,
        "prediction": d.prediction,
        "market_price": (
            str(price_cc_to_decimal(d.market_price_cc)) if d.market_price_cc else None
        ),
        "fair_value": str(price_cc_to_decimal(d.fair_value_cc)) if d.fair_value_cc else None,
        "edge_cc": d.edge_cc,
        "confidence": d.confidence,
        "expected_value_usd": (
            str(micros_to_decimal(d.expected_value_micros))
            if d.expected_value_micros is not None
            else None
        ),
        "side": d.side,
        "requested": str(qc_to_decimal(d.requested_qty_qc or 0)),
        "approved": str(qc_to_decimal(d.approved_qty_qc or 0)),
        "decision": d.decision,
        "reason_code": d.reason_code,
        "reason": d.reason,
        "order_id": d.order_id,
        "projected_leverage": (d.leverage_assessment or {}).get("projected_leverage"),
        "environment": d.environment,
    }


@api.get("/risk/events")
async def risk_events(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    db = get_database()
    async with db.session() as session:
        rows = (
            await session.execute(select(RiskEvent).order_by(desc(RiskEvent.ts)).limit(limit))
        ).scalars().all()
    return {
        "events": [
            {
                "ts": r.ts.isoformat() if r.ts else None,
                "market_type": r.market_type,
                "ticker": r.ticker,
                "strategy_id": r.strategy_id,
                "reason_code": r.reason_code,
                "message": r.message,
                "severity": r.severity,
                "limit_name": r.limit_name,
                "limit_value": r.limit_value,
                "observed_value": r.observed_value,
                "decision_id": r.decision_id,
            }
            for r in rows
        ]
    }


@api.get("/risk/limits")
async def risk_limits() -> dict[str, Any]:
    """Every risk limit, its current utilisation and remaining capacity."""
    ctx = get_context()
    if ctx.app is None:
        raise HTTPException(status_code=503, detail="application is not running")
    settings = ctx.app.settings
    out: dict[str, Any] = {
        "perp_leverage": {
            "limit": f"{MAX_PERP_LEVERAGE:.2f}x",
            "configurable": False,
            "note": "architectural invariant; not a setting and cannot be changed",
            **ctx.leverage_panel(),
        },
        "event": settings.event_risk.model_dump(mode="json"),
        "perp": settings.perp_risk.model_dump(mode="json"),
    }
    for domain, loop in ctx.loops.items():
        state = loop.account_state
        if state is None:
            continue
        key = f"{domain.value}_utilisation"
        if domain is MarketType.EVENT:
            out[key] = {
                "exposure_usd": str(micros_to_decimal(state.total_exposure_micros())),
                "equity_usd": str(micros_to_decimal(state.equity_micros)),
                "open_positions": sum(1 for p in state.positions.values() if p.position_qc != 0),
                "drawdown_pct": str(state.drawdown_pct()),
            }
        else:
            out[key] = {
                "exposure_usd": str(micros_to_decimal(state.effective_exposure_micros())),
                "equity_usd": str(micros_to_decimal(state.equity_micros)),
                "open_positions": sum(1 for p in state.positions.values() if p.position_qc != 0),
            }
    return out


@api.get("/audit")
async def audit(
    action: str | None = None,
    hours: int = Query(24, ge=1, le=8760),
    limit: int = Query(500, ge=1, le=2000),
) -> dict[str, Any]:
    db = get_database()
    since = datetime.now(UTC) - timedelta(hours=hours)
    query = (
        select(AuditLog).where(AuditLog.ts >= since).order_by(desc(AuditLog.ts)).limit(limit)
    )
    if action:
        query = query.where(AuditLog.action == action)
    async with db.session() as session:
        rows = (await session.execute(query)).scalars().all()
    return {
        "entries": [
            {
                "ts": a.ts.isoformat() if a.ts else None,
                "actor": a.actor,
                "action": a.action,
                "subject": a.subject,
                "detail": a.detail,
                "severity": a.severity,
                "environment": a.environment,
            }
            for a in rows
        ]
    }


# =============================================================================
# Strategies, models, research
# =============================================================================


@api.get("/strategies")
async def strategies() -> dict[str, Any]:
    """Registered strategies with per-stage performance kept separate."""
    ctx = get_context()
    live: dict[str, Any] = {}
    for loop in ctx.loops.values():
        for s in loop.strategies:
            live[s.strategy_id] = {
                "state": s.state.value,
                "allocated_capital_usd": str(
                    micros_to_decimal(s.allocated_capital_micros)
                ),
                "metadata": s.metadata.to_dict(),
                "health": s.health(),
            }

    db = get_database()
    async with db.session() as session:
        rows = (await session.execute(select(Strategy))).scalars().all()
    persisted = [
        {
            "strategy_id": s.strategy_id,
            "version": s.version,
            "name": s.name,
            "market_type": s.market_type,
            "family": s.family,
            "state": s.state,
            "enabled": s.enabled,
            "confidence": s.confidence,
            "allocated_capital_usd": str(micros_to_decimal(s.allocated_capital_micros)),
            # Never blended: each stage is reported on its own.
            "train_metrics": s.train_metrics,
            "validation_metrics": s.validation_metrics,
            "oos_metrics": s.oos_metrics,
            "walk_forward_metrics": s.walk_forward_metrics,
            "robustness_metrics": s.robustness_metrics,
            "paper_metrics": s.paper_metrics,
            "live_metrics": s.live_metrics,
            "retire_reason": s.retire_reason,
            "runtime": live.get(s.strategy_id),
        }
        for s in rows
    ]
    # Include runtime-only strategies not yet persisted.
    known = {s["strategy_id"] for s in persisted}
    for sid, info in live.items():
        if sid not in known:
            persisted.append({"strategy_id": sid, "runtime": info, "state": info["state"]})
    return {"strategies": persisted}


@api.get("/strategies/{strategy_id}/history")
async def strategy_history(strategy_id: str) -> dict[str, Any]:
    db = get_database()
    async with db.session() as session:
        rows = (
            await session.execute(
                select(StrategyStateChange)
                .where(StrategyStateChange.strategy_id == strategy_id)
                .order_by(desc(StrategyStateChange.ts))
            )
        ).scalars().all()
    return {
        "strategy_id": strategy_id,
        "transitions": [
            {
                "ts": r.ts.isoformat() if r.ts else None,
                "from": r.from_state,
                "to": r.to_state,
                "actor": r.actor,
                "reason": r.reason,
                "gate_results": r.gate_results,
            }
            for r in rows
        ],
    }


@api.get("/models")
async def models() -> dict[str, Any]:
    db = get_database()
    async with db.session() as session:
        rows = (
            await session.execute(
                select(ModelRegistry).order_by(desc(ModelRegistry.created_at)).limit(200)
            )
        ).scalars().all()
    return {
        "models": [
            {
                "model_id": m.model_id,
                "version": m.version,
                "name": m.name,
                "model_type": m.model_type,
                "market_type": m.market_type,
                "dataset_id": m.dataset_id,
                "feature_set_id": m.feature_set_id,
                "train_window": [
                    m.train_start.isoformat() if m.train_start else None,
                    m.train_end.isoformat() if m.train_end else None,
                ],
                "validation_window": [
                    m.validation_start.isoformat() if m.validation_start else None,
                    m.validation_end.isoformat() if m.validation_end else None,
                ],
                "test_window": [
                    m.test_start.isoformat() if m.test_start else None,
                    m.test_end.isoformat() if m.test_end else None,
                ],
                "train_metrics": m.train_metrics,
                "validation_metrics": m.validation_metrics,
                "test_metrics": m.test_metrics,
                "live_metrics": m.live_metrics,
                "calibration": m.calibration,
                "deployment_state": m.deployment_state,
                "n_train_rows": m.n_train_rows,
                "git_commit": m.git_commit,
            }
            for m in rows
        ]
    }


@api.get("/news")
async def news(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    db = get_database()
    async with db.session() as session:
        rows = (
            await session.execute(
                select(NewsArticle).order_by(desc(NewsArticle.published_ts)).limit(limit)
            )
        ).scalars().all()
        links = (
            await session.execute(
                select(NewsMarketLink).order_by(desc(NewsMarketLink.ts)).limit(limit * 3)
            )
        ).scalars().all()
    by_article: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        by_article.setdefault(link.article_id, []).append(
            {
                "ticker": link.ticker,
                "relevance": link.relevance,
                "estimated_impact": link.estimated_impact,
                "direction": link.direction,
                "confidence": link.confidence,
            }
        )
    return {
        "articles": [
            {
                "article_id": a.article_id,
                "title": a.title,
                "summary": a.summary,
                "url": a.url,
                "source": a.source,
                "published_ts": a.published_ts.isoformat() if a.published_ts else None,
                # Both timestamps: the gap is what a backtest must respect.
                "ingested_ts": a.ingested_ts.isoformat() if a.ingested_ts else None,
                "source_reliability": a.source_reliability,
                "relevance": a.relevance,
                "entities": a.entities,
                "extracted_events": a.extracted_events,
                "markets": by_article.get(a.article_id, []),
            }
            for a in rows
        ]
    }


@api.get("/arbitrage")
async def arbitrage(
    actioned: bool | None = None, limit: int = Query(100, ge=1, le=500)
) -> dict[str, Any]:
    """Detected opportunities, including declined ones and why."""
    db = get_database()
    query = (
        select(ArbitrageOpportunity)
        .order_by(desc(ArbitrageOpportunity.ts))
        .limit(limit)
    )
    if actioned is not None:
        query = query.where(ArbitrageOpportunity.actioned == actioned)
    async with db.session() as session:
        rows = (await session.execute(query)).scalars().all()
    return {
        "opportunities": [
            {
                "opportunity_id": o.opportunity_id,
                "ts": o.ts.isoformat() if o.ts else None,
                "kind": o.kind,
                "market_type": o.market_type,
                "legs": o.legs,
                "gross_edge_usd": _usd(o.gross_edge_micros),
                "fees_usd": _usd(o.fees_micros),
                "spread_cost_usd": _usd(o.spread_cost_micros),
                "slippage_usd": _usd(o.slippage_micros),
                "net_edge_usd": _usd(o.net_edge_micros),
                "expected_value_usd": _usd(o.expected_value_micros),
                "worst_case_usd": _usd(o.worst_case_micros),
                "execution_probability": o.execution_probability,
                "actioned": o.actioned,
                "reason": o.reason,
            }
            for o in rows
        ]
    }


@api.get("/llm/signals")
async def llm_signals(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    """Every LLM consultation, used or not. The LLM has no order authority."""
    db = get_database()
    async with db.session() as session:
        rows = (
            await session.execute(select(LLMSignal).order_by(desc(LLMSignal.ts)).limit(limit))
        ).scalars().all()
    return {
        "signals": [
            {
                "signal_id": s.signal_id,
                "ts": s.ts.isoformat() if s.ts else None,
                "provider": s.provider,
                "model": s.model,
                "ticker": s.ticker,
                "probability": s.probability,
                "confidence": s.confidence,
                "accepted": s.accepted,
                "rejection_reason": s.rejection_reason,
                "decision_id": s.decision_id,
                "latency_ms": s.latency_ms,
                "parsed_output": s.parsed_output,
            }
            for s in rows
        ]
    }


def _usd(micros: int | None) -> str | None:
    return str(micros_to_decimal(micros)) if micros is not None else None


# =============================================================================
# Controls
# =============================================================================


class ModeRequest(BaseModel):
    mode: OperatingMode
    actor: str = "dashboard"


class KillSwitchRequest(BaseModel):
    scope: KillSwitchScope
    reason: str = Field(min_length=1, max_length=500)
    subject: str = ""
    actor: str = "dashboard"


class ReleaseRequest(BaseModel):
    scope: KillSwitchScope
    subject: str = ""
    actor: str = "dashboard"


@api.post("/control/mode")
async def set_mode(request: ModeRequest) -> dict[str, Any]:
    """Change autonomy mode. Affects no risk limit."""
    await get_context().set_operating_mode(request.mode, actor=request.actor)
    return {"operating_mode": request.mode.value}


@api.get("/docs")
async def docs() -> dict[str, Any]:
    """Every explanation the interface shows, with live values.

    Served from the engine rather than baked into the dashboard so a limit's
    documented value is the one actually in force, and so the wording cannot
    drift between the tooltip, the limits table and the docs page.
    """
    ctx = get_context()
    settings = ctx.app.settings if ctx.app else None
    return build_docs(settings)


@api.get("/risk/tunables")
async def risk_tunables() -> dict[str, Any]:
    """Every adjustable risk limit, plus the ones that are not adjustable.

    Locked entries are returned in the same list rather than omitted, so the
    dashboard shows the leverage ceiling beside the sliders instead of leaving a
    reader to wonder where it went.
    """
    ctx = get_context()
    if ctx.app is None:
        raise HTTPException(status_code=503, detail="application is not running")
    return {"tunables": current_values(ctx.app.settings)}


class TunableRequest(BaseModel):
    key: str
    value: str | float | int
    actor: str = "dashboard"


@api.post("/risk/tunables")
async def set_risk_tunable(request: TunableRequest) -> dict[str, Any]:
    """Adjust one risk limit. Applied live, audited, and persisted.

    A refusal returns 400 with the reason, because "that value is outside the
    permitted range" is something the operator needs to read, not a generic
    failure.
    """
    try:
        return await get_context().set_risk_tunable(
            request.key, request.value, actor=request.actor
        )
    except TunableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@api.post("/control/pause")
async def pause(actor: str = "dashboard") -> dict[str, Any]:
    await get_context().pause_trading(actor=actor)
    return {"paused": True}


@api.post("/control/resume")
async def resume(actor: str = "dashboard") -> dict[str, Any]:
    await get_context().resume_trading(actor=actor)
    return {"paused": False}


@api.post("/control/kill-switch/engage")
async def engage_kill_switch(request: KillSwitchRequest) -> dict[str, Any]:
    await get_context().engage_kill_switch(
        request.scope, reason=request.reason, actor=request.actor, subject=request.subject
    )
    return {"scope": request.scope.value, "engaged": True}


@api.post("/control/kill-switch/release")
async def release_kill_switch(request: ReleaseRequest) -> dict[str, Any]:
    await get_context().release_kill_switch(
        request.scope, actor=request.actor, subject=request.subject
    )
    return {"scope": request.scope.value, "engaged": False}


@api.post("/control/cancel-all")
async def cancel_all(
    market_type: Literal["event", "perp"] | None = None,
) -> dict[str, Any]:
    """Cancel every working order. Always permitted -- it only reduces risk."""
    domain = MarketType(market_type) if market_type else None
    cancelled = await get_context().cancel_all_orders(domain)
    return {"cancelled": cancelled}


@api.get("/control/state")
async def control_state() -> dict[str, Any]:
    """What the control panel should render as current."""
    ctx = get_context()
    if ctx.app is None:
        raise HTTPException(status_code=503, detail="application is not running")
    return {
        "operating_mode": ctx.app.settings.operating_mode.value,
        "environment": ctx.app.settings.environment.value,
        "live_armed": ctx.app.settings.live_trading_armed,
        "can_submit_real_orders": ctx.app.can_submit_real_orders,
        "trading_permitted": ctx.app.trading_permitted,
        "phase": ctx.app.phase.value,
        "max_perp_leverage": f"{MAX_PERP_LEVERAGE:.2f}",
        "kill_switches": [
            {
                "scope": s.scope,
                "subject": s.subject,
                "engaged": s.engaged,
                "reason": s.reason,
                "engaged_by": s.engaged_by,
            }
            for s in ctx.app.kill_switches.all_statuses()
        ],
        "modes": [m.value for m in OperatingMode],
        "scopes": [s.value for s in KillSwitchScope],
    }


app.include_router(api)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "Kalshi Autonomous Trader",
        "version": "0.1.0",
        "max_perp_leverage": f"{MAX_PERP_LEVERAGE:.2f}",
        "docs": "/docs",
    }


def main() -> None:
    """Run the API standalone, for development against an idle engine."""
    import uvicorn

    from kat.config.settings import get_settings

    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")


# Referenced by /markets to avoid an unused-import warning on Market.
_ = (Market, func)
