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
from kat.common.fixedpoint import (
    CC_PER_DOLLAR,
    QC_PER_CONTRACT,
    micros_to_decimal,
)
from kat.config.settings import EventRiskSettings
from kat.kalshi.fees import estimate_fee_micros, prorate_fee_micros
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger
from kat.observability.metrics import M_RISK_REJECTIONS, metrics
from kat.polymarket.fees import PolymarketFeeSchedule, polymarket_fee_micros
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
    #: When this position was opened. A strategy trading a forecast with a
    #: horizon has to know how much of that horizon has elapsed, and the
    #: backtester's own ledger has tracked this from the start.
    opened_at_ms: int = 0

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


    def apply_fill(
        self,
        *,
        ticker: str,
        qty_qc: int,
        price_cc: int,
        is_buy: bool,
        fee_micros: int = 0,
        event_ticker: str = "",
        now_ms: int = 0,
    ) -> int:
        """Apply one fill to this account, returning the realised P&L in micros.

        In live and demo the reconciler rebuilds positions from the exchange, so
        nothing local has to. Paper has no exchange to ask: the fills are
        simulated, and without this the account never changed. 108 fills were
        recorded against an account that still read exactly its seeded $200 with
        no positions, no exposure and no profit or loss -- so there was no way to
        tell a strategy that was working from one that was not.

        Semantics follow ``position_qc`` as Kalshi reports it: positive is YES,
        negative is NO. Selling YES you do not hold *is* buying NO, so a sell
        that opens a position costs ``1 - price`` per contract, and a position is
        closed only when a fill opposes the sign it already has.
        """
        position = self.positions.get(ticker)
        if position is None:
            position = EventPositionState(ticker=ticker, event_ticker=event_ticker)
            self.positions[ticker] = position
        if position.position_qc == 0:
            # Opening from flat, so the clock on this position starts now. Adding
            # to an existing position leaves it alone: the horizon belongs to the
            # forecast that opened it.
            position.opened_at_ms = now_ms
        if event_ticker and not position.event_ticker:
            position.event_ticker = event_ticker

        # The fee is paid whichever way the fill goes, opening or closing.
        self.balance_micros -= fee_micros

        direction = 1 if is_buy else -1
        remaining = qty_qc
        realised = 0

        closing = 0
        if position.position_qc != 0 and (position.position_qc > 0) != (direction > 0):
            closing = min(remaining, abs(position.position_qc))

        if closing:
            held = abs(position.position_qc)
            long_position = position.position_qc > 0
            # A long holds YES and is worth the YES price; a short holds NO and
            # is worth what is left of the dollar.
            proceeds_cc = price_cc if long_position else CC_PER_DOLLAR - price_cc
            proceeds = proceeds_cc * closing

            basis_total = (position.avg_entry_cc or 0) * held
            released = basis_total if closing == held else basis_total * closing // held

            realised = proceeds - released
            self.balance_micros += proceeds
            self.realized_pnl_today_micros += realised
            position.position_qc -= closing if long_position else -closing
            if position.position_qc == 0:
                position.avg_entry_cc = None
            remaining -= closing

        if remaining > 0:
            # Opening or adding. Buying YES costs the YES price; selling to open
            # is buying NO, which costs the rest of the dollar.
            cost_cc = price_cc if direction > 0 else CC_PER_DOLLAR - price_cc
            held = abs(position.position_qc)
            basis_total = (position.avg_entry_cc or 0) * held + cost_cc * remaining
            position.position_qc += direction * remaining
            new_held = abs(position.position_qc)
            position.avg_entry_cc = (basis_total // new_held) if new_held else None
            self.balance_micros -= cost_cc * remaining

        # Exposure is what the position could still lose, which for an event
        # contract is exactly what it cost: the downside is the whole premium.
        position.exposure_micros = (position.avg_entry_cc or 0) * abs(position.position_qc)

        if position.position_qc == 0:
            self.positions.pop(ticker, None)

        if now_ms:
            self.updated_at_ms = now_ms
        self.peak_equity_micros = max(self.peak_equity_micros, self.equity_micros)
        return realised

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

    def __init__(
        self,
        settings: EventRiskSettings,
        *,
        max_state_age_ms: int = 30_000,
        polymarket_fees: PolymarketFeeSchedule | None = None,
    ) -> None:
        self.settings = settings
        self.max_state_age_ms = max_state_age_ms
        #: The venue's own schedule, when this engine is judging orders for one
        #: that is not Kalshi. Supplied by whoever builds the engine, because an
        #: intent does not say which exchange it is bound for and this engine
        #: must not take a strategy's word for what its costs are.
        self.polymarket_fees = polymarket_fees

    def _passive_fee_cc(self, price_cc: int) -> int:
        """What one resting contract's fee costs here, in centicents. Signed."""
        if self.polymarket_fees is not None:
            return polymarket_fee_micros(
                price_cc=price_cc,
                qty_qc=QC_PER_CONTRACT,
                schedule=self.polymarket_fees,
                is_taker=False,
            ) // 100
        return estimate_fee_micros(
            price_cc=price_cc, qty_qc=QC_PER_CONTRACT
        ).fee_micros // 100

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
        # Not ``intent.reduce_only or ...``. That flag is set by the strategy, and
        # this engine exists precisely so a strategy cannot authorise its own
        # risk; an intent asserting reduce_only with no position, or with a size
        # far past it, would have skipped every check below on its own say-so.
        # The flag still travels to the exchange on the order, where it is
        # enforced by something that is not us. What earns the exemption here is
        # the position we can see and the size of the order against it.
        is_reducing = self._is_reducing(intent, position)

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
            #
            # Whether this order would open a position, not whether the market has
            # an entry in the dict. Entries exist for markets with only a resting
            # order, and used to outlive it; the membership test then exempted any
            # market ever quoted from this cap for good. Reproduced: an account at
            # the cap rejected a new market with MAX_OPEN_POSITIONS and approved
            # the same order on a market that had once held a quote.
            held = state.positions.get(intent.ticker)
            if held is None or held.position_qc == 0:
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
                # A resting order is not paying the spread, it is being paid it.
                #
                # min_edge_cc asks whether an edge justifies crossing, and for a
                # taker the spread is the largest part of what crossing costs. A
                # post-only order never crosses: its edge *is* the half-spread it
                # will earn, so requiring that half to clear a bar sized for the
                # whole thing demands a four-cent spread to justify earning two.
                # It blocked exactly the trades whose economics are best -- the
                # only structure measured on this venue that is positive before
                # any forecast.
                #
                # The floor becomes one fee, because this intent is one leg. A
                # passive buy at the bid earns half the spread and pays one fee;
                # the second fee arrives with the second leg, which earns its own
                # half. Charging both against one half is the same double-count
                # in a different place, and it put the bar at twice what the
                # strategy's own economics require.
                #
                # The padded estimate rather than the realised fee, because this
                # is a gate: an unknown schedule must not make a marginal trade
                # look affordable. min_expected_value_micros below still does the
                # economic work, computed net of a full round trip either way.
                floor = self.settings.min_edge_cc
                if intent.post_only:
                    # Never below zero, whatever the venue pays. A rebate makes
                    # this fee negative, and a negative floor would admit a
                    # quote with no edge at all on the strength of a third of a
                    # centicent. The floor exists to stop a trade that cannot
                    # cover its own fee; where there is no fee to cover it stops
                    # nothing, and min_expected_value_micros below does the
                    # economic work.
                    #
                    # Charging Kalshi's fee here regardless was worth measuring:
                    # on Polymarket it refused 47,834 intents in a day, and
                    # every one of them was refused for a cost the venue does
                    # not levy.
                    floor = max(0, self._passive_fee_cc(intent.limit_price_cc))
                ok = edge >= floor
                checks.append(
                    CheckResult("min_edge", ok,
                                limit=str(floor), observed=str(edge),
                                detail="passive: fee floor" if intent.post_only else None)
                )
                if not ok:
                    return self._rejected(
                        intent, "INSUFFICIENT_EDGE",
                        "estimated edge does not clear the minimum",
                        checks,
                    )

            if intent.expected_value_micros is not None:
                # Compared directly, not after subtracting fees again.
                #
                # OrderIntent.expected_value_micros is already net: the strategy
                # builds it from Costing.net_micros, which is gross minus a
                # *round-trip* fee. Subtracting this engine's entry-fee estimate
                # on top charged the trade twice, and the second charge was not
                # even the same quantity -- one leg against two. Marginal trades
                # whose real expected value was positive were rejected, which is
                # invisible in a log: a refusal looks identical either way.
                net_ev = intent.expected_value_micros
                ok = net_ev >= self.settings.min_expected_value_micros
                checks.append(
                    CheckResult("min_expected_value", ok,
                                limit=str(micros_to_decimal(self.settings.min_expected_value_micros)),
                                observed=str(micros_to_decimal(net_ev)),
                                detail="net of the round-trip fees the strategy costed")
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
            # Restated for the size actually approved. The caller costed the
            # quantity the strategy asked for, and this engine routinely approves
            # a fraction of it, so passing the original figure through reports a
            # fee for an order that was never placed.
            estimated_fees_micros=prorate_fee_micros(
                estimated_fees_micros,
                from_qty_qc=intent.qty_qc,
                to_qty_qc=approved_qty,
            ),
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
        """True when this order *can only* move an existing position toward flat.

        Size is part of the question, not just direction. Being classified as
        reducing exempts an order from every check in sections 2 and 3 -- the
        daily-loss and drawdown halts, the order-rate limit, available capital,
        max loss per trade, the per-market and portfolio exposure caps, the
        correlated-event cap, the per-order notional cap, the open-position
        count, minimum edge and minimum expected value. Only the freshness gate
        survives. That exemption is right for an exit: closing a position must
        never be blocked by the limits that govern opening one, or a halted
        platform could not get out of the trade that halted it.

        It was previously granted on direction alone, so *any* opposite-side buy
        qualified however large. One contract of residual YES bought an unbounded
        NO position with nothing checked: measured, an account holding a single
        contract with half a cent of equity, $100 of losses on the day, a
        near-total drawdown and ten thousand orders in the minute was approved
        for 5,000 contracts -- $2,500 of notional against a $25 per-order cap --
        with one check recorded. This is reachable from the shipped strategies
        rather than hypothetical: ``headroom_qc(..., increases=False)`` returns
        ``cap + position_qc`` precisely so a reversing signal can close and
        re-open in one intent.

        An order larger than the position it opposes is a reversal: part of it
        closes, and the rest opens new risk on the other side. That is an entry,
        and it faces what entries face. The whole order is then measured rather
        than only its excess, which slightly overstates the new risk -- the safe
        direction, and far simpler than splitting one intent into two verdicts.
        """
        if position.position_qc == 0:
            return False
        holding_yes = position.position_qc > 0
        buying_yes = intent.outcome_side is OutcomeSide.YES
        if holding_yes == buying_yes:
            # Same side: this adds to the position.
            return False
        # Buying the opposite outcome offsets the position, but only up to the
        # size actually held.
        return intent.qty_qc <= abs(position.position_qc)

    @staticmethod
    def _edge_cc(intent: OrderIntent) -> int:
        """Edge in centicents, from the single definition on the intent.

        This gate and the audit record used to compute it separately, and only
        one of them signed the NO side correctly.
        """
        return intent.edge_cc() or 0

    def _check_market_quality(
        self, book: OrderBook, qty: int, intent: OrderIntent, checks: list[CheckResult]
    ) -> tuple[bool, int | None]:
        if book.is_crossed:
            checks.append(
                CheckResult("book_integrity", False, detail="book is crossed; data is unreliable")
            )
            return False, None
        checks.append(CheckResult("book_integrity", True))

        if intent.post_only:
            # The three checks below are taker guards, and every one of them
            # inverts for an order that rests: a wide spread is what a maker is
            # paid, depth on the far side is what a taker fills against while a
            # maker joins the near side and waits, and slippage is the cost of
            # crossing, which a post-only order cannot do.
            #
            # Leaving them did not make the platform safer, it made it
            # contradictory. The strategy requires four cents of spread to cover
            # adverse selection and this check refused anything above three, so
            # between them they approved nothing at all: 34,846 intents in a day
            # and not one order.
            #
            # What still bounds the loss is unchanged -- position, exposure,
            # correlated exposure, notional and open-position limits are all
            # above this point and all still apply.
            checks.append(
                CheckResult(
                    "taker_guards_skipped", True,
                    detail="post-only: spread, far-side depth and slippage are "
                           "costs of crossing, and this order rests",
                )
            )
            return True, None

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
