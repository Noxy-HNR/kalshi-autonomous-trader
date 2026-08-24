"""Event-contract risk engine.

Entirely separate from :mod:`kat.risk.perp_risk`. They share the vocabulary in
:mod:`kat.risk.types` and nothing else, because the two products have different
risk in a way that a shared calculation would have to paper over:

* An event contract is **fully collateralised**. Buying YES at $0.40 costs $0.40
  and can lose at most $0.40. There is no margin, no mark price, no funding, no
  liquidation, and therefore no leverage to constrain -- the maximum loss of a
  position is known exactly at entry.
* The binding constraints are instead **capital**, **concentration** and **loss
  budgets**: you cannot spend money you do not have, you should not put it all
  in one event, and you should stop for the day after losing enough.

Because maximum loss is known exactly, position sizing here is honest in a way
it cannot be for a marked product: ``max_loss = qty * entry_price`` for a long,
and ``qty * (1 - entry_price)`` for the other side. No volatility estimate is
required to bound the downside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from kat.common.enums import MarketType, OutcomeSide, RiskDecision
from kat.common.fixedpoint import CC_PER_DOLLAR, micros_to_decimal
from kat.config.settings import EventRiskSettings
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger
from kat.observability.metrics import M_RISK_REJECTIONS, metrics
from kat.risk.types import CheckResult, OrderIntent, RiskVerdict

log = get_logger(__name__)

__all__ = ["EventAccountState", "EventPositionState", "EventRiskEngine"]


@dataclass(slots=True)
class EventPositionState:
    """Our view of one event-contract position.

    ``position_qc`` is signed the way Kalshi reports it: positive is YES,
    negative is NO.
    """

    ticker: str
    event_ticker: str = ""
    position_qc: int = 0
    avg_entry_cc: int | None = None
    # Capital actually committed to this market.
    exposure_micros: int = 0
    pending_exposure_micros: int = 0
    unrealized_pnl_micros: int = 0

    @property
    def max_loss_micros(self) -> int:
        """Worst case for this position: the whole committed cost."""
        return self.exposure_micros


@dataclass(slots=True)
class EventAccountState:
    """Account state for the event-contract domain."""

    balance_micros: int = 0            # settled cash available to spend
    portfolio_value_micros: int = 0    # mark value of open positions
    positions: dict[str, EventPositionState] = field(default_factory=dict)

    updated_at_ms: int = 0
    realized_pnl_today_micros: int = 0
    peak_equity_micros: int = 0
    orders_this_minute: int = 0

    @property
    def equity_micros(self) -> int:
        return self.balance_micros + self.portfolio_value_micros

    def total_exposure_micros(self) -> int:
        return sum(p.exposure_micros for p in self.positions.values())

    def total_pending_micros(self) -> int:
        return sum(p.pending_exposure_micros for p in self.positions.values())

    def event_exposure_micros(self, event_ticker: str) -> int:
        """Capital committed across every market in one event.

        This is the concentration measure that matters for event contracts: the
        markets within an event resolve off the same real-world outcome, so
        holding five of them is one bet, not five.
        """
        if not event_ticker:
            return 0
        return sum(
            p.exposure_micros
            for p in self.positions.values()
            if p.event_ticker == event_ticker
        )

    def drawdown_pct(self) -> Decimal:
        if self.peak_equity_micros <= 0:
            return Decimal(0)
        drop = self.peak_equity_micros - self.equity_micros
        if drop <= 0:
            return Decimal(0)
        return (Decimal(drop) / Decimal(self.peak_equity_micros)).quantize(Decimal("0.0001"))


class EventRiskEngine:
    """Independent risk engine for event contracts."""

    def __init__(self, settings: EventRiskSettings, *, max_state_age_ms: int = 30_000) -> None:
        self.settings = settings
        self.max_state_age_ms = max_state_age_ms

    def evaluate(
        self,
        intent: OrderIntent,
        state: EventAccountState,
        *,
        book: OrderBook | None = None,
        now_ms: int,
        event_ticker: str = "",
        estimated_fees_micros: int = 0,
    ) -> RiskVerdict:
        """Decide whether an event-contract order may be sent, and at what size."""
        if intent.market_type is not MarketType.EVENT:
            return RiskVerdict.reject(
                intent, "WRONG_DOMAIN",
                f"event risk engine received a {intent.market_type.value} intent",
            )

        checks: list[CheckResult] = []

        # --- 1. freshness ---------------------------------------------------
        age_ms = now_ms - state.updated_at_ms if state.updated_at_ms else None
        if age_ms is None or age_ms > self.max_state_age_ms:
            checks.append(
                CheckResult("account_state_freshness", False,
                            limit=f"<= {self.max_state_age_ms}ms",
                            observed=f"{age_ms}ms" if age_ms is not None else "never updated")
            )
            return self._rejected(
                intent, "STALE_ACCOUNT_STATE",
                "event-contract account state is stale; refusing to size an order from it",
                checks,
            )
        checks.append(CheckResult("account_state_freshness", True, observed=f"{age_ms}ms"))

        position = state.positions.get(intent.ticker) or EventPositionState(
            ticker=intent.ticker, event_ticker=event_ticker
        )
        is_reducing = intent.reduce_only or self._is_reducing(intent, position)

        # --- 2. loss budgets: these halt trading, not just resize it --------
        if not is_reducing:
            equity = state.equity_micros
            if equity <= 0:
                checks.append(CheckResult("account_equity", False, observed="0"))
                return self._rejected(
                    intent, "NO_EQUITY", "no event-contract equity available", checks
                )

            daily_limit = int(Decimal(equity) * self.settings.max_daily_loss_pct)
            daily_loss = max(0, -state.realized_pnl_today_micros)
            ok = daily_loss < daily_limit
            checks.append(
                CheckResult("max_daily_loss", ok,
                            limit=str(micros_to_decimal(daily_limit)),
                            observed=str(micros_to_decimal(daily_loss)))
            )
            if not ok:
                return self._rejected(
                    intent, "DAILY_LOSS_LIMIT",
                    "daily loss limit reached; no new event-contract risk today",
                    checks,
                )

            dd = state.drawdown_pct()
            dd_ok = dd < self.settings.max_drawdown_pct
            checks.append(
                CheckResult("max_drawdown", dd_ok,
                            limit=f"{self.settings.max_drawdown_pct:.2%}",
                            observed=f"{dd:.2%}")
            )
            if not dd_ok:
                return self._rejected(
                    intent, "DRAWDOWN_LIMIT",
                    "drawdown limit reached; capital preservation mode",
                    checks,
                )

            rate_ok = state.orders_this_minute < self.settings.max_orders_per_minute
            checks.append(
                CheckResult("order_rate", rate_ok,
                            limit=str(self.settings.max_orders_per_minute),
                            observed=str(state.orders_this_minute))
            )
            if not rate_ok:
                return self._rejected(
                    intent, "ORDER_RATE_LIMIT",
                    "order rate limit reached; runaway-order protection engaged",
                    checks,
                )

        # --- 3. sizing ------------------------------------------------------
        approved_qty = intent.qty_qc
        cost_per_contract_cc = self._cost_per_contract_cc(intent)
        if cost_per_contract_cc <= 0:
            checks.append(CheckResult("valid_price", False, observed=str(intent.limit_price_cc)))
            return self._rejected(
                intent, "INVALID_PRICE",
                "order price implies a zero or negative cost per contract",
                checks,
            )

        if not is_reducing:
            # 3a. Cannot spend money we do not have. Fees are included, because
            # an order that fills and then cannot pay its fee is not affordable.
            spendable = state.balance_micros - state.total_pending_micros() - estimated_fees_micros
            if spendable <= 0:
                checks.append(
                    CheckResult("available_capital", False,
                                observed=str(micros_to_decimal(max(0, spendable))))
                )
                return self._rejected(
                    intent, "INSUFFICIENT_CAPITAL",
                    "no uncommitted cash remains for a new event-contract position",
                    checks,
                )
            affordable_qty = spendable // cost_per_contract_cc
            if affordable_qty < approved_qty:
                approved_qty = affordable_qty
            checks.append(
                CheckResult("available_capital", approved_qty > 0,
                            limit=str(micros_to_decimal(spendable)),
                            observed=str(micros_to_decimal(intent.qty_qc * cost_per_contract_cc)),
                            detail=None if approved_qty == intent.qty_qc else "resized to available cash")
            )
            if approved_qty <= 0:
                return self._rejected(
                    intent, "INSUFFICIENT_CAPITAL",
                    "not enough cash for even one contract at this price",
                    checks,
                )

            # 3b. Maximum loss on this single trade.
            max_loss_limit = int(Decimal(state.equity_micros) * self.settings.max_loss_per_trade_pct)
            max_qty_by_loss = max_loss_limit // cost_per_contract_cc
            if max_qty_by_loss < approved_qty:
                approved_qty = max_qty_by_loss
            checks.append(
                CheckResult("max_loss_per_trade", approved_qty > 0,
                            limit=str(micros_to_decimal(max_loss_limit)),
                            observed=str(micros_to_decimal(approved_qty * cost_per_contract_cc)))
            )
            if approved_qty <= 0:
                return self._rejected(
                    intent, "MAX_LOSS_PER_TRADE",
                    "one contract would already exceed the per-trade loss budget",
                    checks,
                )

            # 3c. Position concentration in this market.
            pos_limit = int(Decimal(state.equity_micros) * self.settings.max_position_pct_equity)
            headroom = pos_limit - position.exposure_micros - position.pending_exposure_micros
            max_qty_by_position = max(0, headroom) // cost_per_contract_cc
            if max_qty_by_position < approved_qty:
                approved_qty = max_qty_by_position
            checks.append(
                CheckResult("max_position_size", approved_qty > 0,
                            limit=str(micros_to_decimal(pos_limit)),
                            observed=str(micros_to_decimal(position.exposure_micros)))
            )
            if approved_qty <= 0:
                return self._rejected(
                    intent, "POSITION_LIMIT",
                    "position limit for this market is already reached",
                    checks,
                )

            # 3d. Total portfolio exposure.
            port_limit = int(
                Decimal(state.equity_micros) * self.settings.max_portfolio_exposure_pct
            )
            used = state.total_exposure_micros() + state.total_pending_micros()
            max_qty_by_portfolio = max(0, port_limit - used) // cost_per_contract_cc
            if max_qty_by_portfolio < approved_qty:
                approved_qty = max_qty_by_portfolio
            checks.append(
                CheckResult("max_portfolio_exposure", approved_qty > 0,
                            limit=str(micros_to_decimal(port_limit)),
                            observed=str(micros_to_decimal(used)))
            )
            if approved_qty <= 0:
                return self._rejected(
                    intent, "PORTFOLIO_EXPOSURE_LIMIT",
                    "total event-contract exposure limit reached",
                    checks,
                )

            # 3e. Correlated exposure: markets in one event are one bet.
            if event_ticker:
                corr_limit = int(
                    Decimal(state.equity_micros) * self.settings.max_correlated_exposure_pct
                )
                event_used = state.event_exposure_micros(event_ticker)
                max_qty_by_corr = max(0, corr_limit - event_used) // cost_per_contract_cc
                if max_qty_by_corr < approved_qty:
                    approved_qty = max_qty_by_corr
                checks.append(
                    CheckResult("max_correlated_exposure", approved_qty > 0,
                                limit=str(micros_to_decimal(corr_limit)),
                                observed=str(micros_to_decimal(event_used)),
                                detail=f"event {event_ticker}")
                )
                if approved_qty <= 0:
                    return self._rejected(
                        intent, "CORRELATED_EXPOSURE_LIMIT",
                        f"correlated exposure limit for event {event_ticker} reached",
                        checks,
                    )

            # 3f. Per-order notional cap.
            max_order_micros = int(self.settings.max_order_notional_usd * 1_000_000)
            if approved_qty * cost_per_contract_cc > max_order_micros:
                approved_qty = max_order_micros // cost_per_contract_cc
            checks.append(
                CheckResult("max_order_notional", approved_qty > 0,
                            limit=f"${self.settings.max_order_notional_usd}",
                            observed=str(micros_to_decimal(approved_qty * cost_per_contract_cc)))
            )
            if approved_qty <= 0:
                return self._rejected(
                    intent, "ORDER_TOO_LARGE",
                    "one contract already exceeds the per-order notional cap",
                    checks,
                )

            # 3g. Open-position count.
            if intent.ticker not in state.positions:
                open_count = sum(1 for p in state.positions.values() if p.position_qc != 0)
                ok = open_count < self.settings.max_open_positions
                checks.append(
                    CheckResult("max_open_positions", ok,
                                limit=str(self.settings.max_open_positions),
                                observed=str(open_count))
                )
                if not ok:
                    return self._rejected(
                        intent, "MAX_OPEN_POSITIONS",
                        f"already holding {open_count} event-contract positions",
                        checks,
                    )

        # --- 4. market quality ----------------------------------------------
        slippage_cc = None
        if book is not None:
            ok, slippage_cc = self._check_market_quality(book, approved_qty, intent, checks)
            if not ok and not is_reducing:
                return self._rejected(
                    intent, "MARKET_QUALITY",
                    "market is too wide, too thin, or would slip too much",
                    checks,
                )

        # --- 5. edge and expected value --------------------------------------
        if not is_reducing:
            if intent.fair_value_cc is not None:
                edge = self._edge_cc(intent)
                ok = edge >= self.settings.min_edge_cc
                checks.append(
                    CheckResult("min_edge", ok,
                                limit=str(self.settings.min_edge_cc), observed=str(edge))
                )
                if not ok:
                    return self._rejected(
                        intent, "INSUFFICIENT_EDGE",
                        "estimated edge does not clear the minimum",
                        checks,
                    )

            if intent.expected_value_micros is not None:
                net_ev = intent.expected_value_micros - estimated_fees_micros
                ok = net_ev >= self.settings.min_expected_value_micros
                checks.append(
                    CheckResult("min_expected_value", ok,
                                limit=str(micros_to_decimal(self.settings.min_expected_value_micros)),
                                observed=str(micros_to_decimal(net_ev)),
                                detail="after estimated fees")
                )
                if not ok:
                    return self._rejected(
                        intent, "INSUFFICIENT_EXPECTED_VALUE",
                        "expected value after costs does not justify the trade",
                        checks,
                    )

        decision = RiskDecision.RESIZED if approved_qty != intent.qty_qc else RiskDecision.APPROVED
        return RiskVerdict(
            decision=decision,
            intent=intent,
            approved_qty_qc=approved_qty,
            reason_code="APPROVED",
            reason=(
                "within all event-contract risk limits"
                if decision is RiskDecision.APPROVED
                else "approved at a reduced size to stay within event-contract risk limits"
            ),
            checks=checks,
            estimated_fees_micros=estimated_fees_micros,
            expected_slippage_cc=slippage_cc,
            current_exposure_micros=state.total_exposure_micros(),
            projected_exposure_micros=(
                state.total_exposure_micros() + approved_qty * cost_per_contract_cc
            ),
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _cost_per_contract_cc(intent: OrderIntent) -> int:
        """Capital committed per contract, which is also the maximum loss.

        Kalshi quotes both sides at the price of the side being bought, so the
        limit price already *is* the cost per contract for a buy. For a sell of
        an existing position no new capital is committed, but the caller only
        reaches sizing logic for opening orders.
        """
        return intent.limit_price_cc

    @staticmethod
    def _is_reducing(intent: OrderIntent, position: EventPositionState) -> bool:
        """True when this order moves an existing position toward flat."""
        if position.position_qc == 0:
            return False
        holding_yes = position.position_qc > 0
        buying_yes = intent.outcome_side is OutcomeSide.YES
        # Buying the opposite outcome offsets the position.
        return holding_yes != buying_yes

    @staticmethod
    def _edge_cc(intent: OrderIntent) -> int:
        """Edge in centicents: how far the price is from our fair value.

        Signed so that a YES buy below fair value and a NO buy above it both
        register as positive edge.
        """
        if intent.fair_value_cc is None:
            return 0
        if intent.outcome_side is OutcomeSide.NO:
            # A NO contract is worth (1 - fair_value); its edge is that value
            # minus what we pay for it.
            return (CC_PER_DOLLAR - intent.fair_value_cc) - intent.limit_price_cc
        return intent.fair_value_cc - intent.limit_price_cc

    def _check_market_quality(
        self, book: OrderBook, qty: int, intent: OrderIntent, checks: list[CheckResult]
    ) -> tuple[bool, int | None]:
        if book.is_crossed:
            checks.append(
                CheckResult("book_integrity", False, detail="book is crossed; data is unreliable")
            )
            return False, None
        checks.append(CheckResult("book_integrity", True))

        spread = book.spread_cc
        spread_ok = spread is not None and spread <= self.settings.max_spread_cc
        checks.append(
            CheckResult("max_spread", spread_ok, limit=str(self.settings.max_spread_cc),
                        observed=str(spread) if spread is not None else "one-sided book")
        )
        if not spread_ok:
            return False, None

        depth = book.total_depth_qc(is_bid=not intent.is_buy)
        depth_ok = depth >= self.settings.min_book_depth_qc
        checks.append(
            CheckResult("min_book_depth", depth_ok,
                        limit=str(self.settings.min_book_depth_qc), observed=str(depth))
        )
        if not depth_ok:
            return False, None

        slippage = book.slippage_cc(qty, is_buy=intent.is_buy)
        slip_ok = slippage is not None and slippage <= self.settings.max_slippage_cc
        checks.append(
            CheckResult("max_slippage", slip_ok, limit=str(self.settings.max_slippage_cc),
                        observed=str(slippage) if slippage is not None else "insufficient depth")
        )
        return slip_ok, slippage

    @staticmethod
    def _rejected(
        intent: OrderIntent, code: str, reason: str, checks: list[CheckResult]
    ) -> RiskVerdict:
        metrics.increment(M_RISK_REJECTIONS, domain="event", reason=code)
        return RiskVerdict.reject(intent, code, reason, checks=checks)
