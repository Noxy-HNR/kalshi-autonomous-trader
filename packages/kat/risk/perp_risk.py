"""Perpetual risk engine.

This is where the 1.00x leverage invariant is actually applied to real orders.
:mod:`kat.risk.invariants` holds the rule; this module supplies it with honest
numbers and refuses to proceed when it cannot.

Computing equity honestly
-------------------------
The denominator of the leverage ratio decides everything, so it is computed
conservatively and its provenance is recorded:

1. Prefer ``account_equity`` from ``GET /margin/balance``. Kalshi documents this
   as 0 for self-clearing members, so a zero is treated as *absent*, never as an
   equity of zero dollars.
2. Fall back to ``settled_funds + position_value``, which is what equity means.
3. If neither is usable, equity is unknown -- and an unknown equity blocks every
   exposure-increasing order. It does not default to anything.

Every parse rounds equity **down** and exposure **up**, so arithmetic error can
only ever make the platform more cautious.

Cross-checking the exchange
---------------------------
``GET /margin/risk`` gives Kalshi's own ``total_position_notional``. We compare
it against our locally computed exposure and use **the larger of the two**. If
our view is somehow too optimistic -- a missed fill, a stale mark -- the
exchange's figure corrects us. We never use the exchange's ``account_leverage``
as the limit, because it is measured against maintenance margin rather than
equity and would permit roughly ten times more real leverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from kat.common.enums import BookSide, MarketType, RiskDecision
from kat.common.errors import LeverageViolation
from kat.common.fixedpoint import micros_to_decimal
from kat.config.settings import PerpRiskSettings
from kat.kalshi.fees import prorate_fee_micros
from kat.kalshi.perps.models import PerpBalance, PerpRiskSnapshot
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger
from kat.observability.metrics import M_LEVERAGE_BLOCKS, M_RISK_REJECTIONS, metrics
from kat.risk.invariants import (
    LeverageAssessment,
    assert_invariant_intact,
    evaluate_perp_leverage,
    position_exposure_micros,
)
from kat.risk.types import CheckResult, OrderIntent, RiskVerdict

log = get_logger(__name__)

__all__ = ["PerpAccountState", "PerpPositionState", "PerpRiskEngine"]


@dataclass(slots=True)
class PerpPositionState:
    """Our live view of one perpetual position."""

    ticker: str
    position_qc: int = 0        # signed: + long, - short
    mark_price_cc: int | None = None
    entry_price_cc: int | None = None
    contract_multiplier_micros: int = 1_000_000
    # Notional that working orders on this market could add if they all filled.
    pending_buy_qc: int = 0
    pending_sell_qc: int = 0

    def exposure_micros(self) -> int:
        """Absolute notional of the filled position."""
        if self.mark_price_cc is None:
            return 0
        base = position_exposure_micros(self.position_qc, self.mark_price_cc)
        if self.contract_multiplier_micros == 1_000_000:
            return base
        # Scale by contract size, rounding up so a fractional multiplier can
        # never understate exposure.
        return -(-base * self.contract_multiplier_micros // 1_000_000)

    def projected_exposure_micros(self, extra_qty_qc: int) -> int:
        """Exposure if ``extra_qty_qc`` (signed) were added to this position."""
        if self.mark_price_cc is None:
            return 0
        new_position = self.position_qc + extra_qty_qc
        base = position_exposure_micros(new_position, self.mark_price_cc)
        if self.contract_multiplier_micros == 1_000_000:
            return base
        return -(-base * self.contract_multiplier_micros // 1_000_000)

    def worst_case_pending_exposure_micros(self) -> int:
        """Exposure added by working orders, taking the worse of the two sides.

        Resting buys and resting sells are not netted: they will not both fill
        at the same instant, and assuming they cancel out would understate the
        exposure we could actually end up holding.
        """
        if self.mark_price_cc is None:
            return 0

        long_case = abs(self.position_qc + self.pending_buy_qc)
        short_case = abs(self.position_qc - self.pending_sell_qc)
        worst_qty = max(long_case, short_case)
        current = abs(self.position_qc)
        added = max(0, worst_qty - current)
        base = added * self.mark_price_cc
        if self.contract_multiplier_micros == 1_000_000:
            return base
        return -(-base * self.contract_multiplier_micros // 1_000_000)


@dataclass(slots=True)
class PerpAccountState:
    """Everything the perpetual risk engine needs to decide.

    Assembled by the reconciliation layer from balance, positions, working
    orders and the exchange's risk endpoint.
    """

    equity_micros: int = 0
    equity_source: str = "unknown"
    equity_is_known: bool = False

    positions: dict[str, PerpPositionState] = field(default_factory=dict)

    # Exchange cross-check figures.
    exchange_total_notional_micros: int = 0
    exchange_account_leverage: float | None = None
    exchange_maintenance_margin_micros: int = 0
    exchange_notional_risk_limit_micros: int | None = None
    liquidation_margin_ratio_threshold: float | None = None

    # Freshness. Stale account state blocks trading.
    updated_at_ms: int = 0

    realized_pnl_today_micros: int = 0
    peak_equity_micros: int = 0

    # -- derived -------------------------------------------------------------

    def total_exposure_micros(self) -> int:
        """Locally computed absolute notional across all positions."""
        return sum(p.exposure_micros() for p in self.positions.values())

    def total_pending_exposure_micros(self) -> int:
        return sum(p.worst_case_pending_exposure_micros() for p in self.positions.values())

    def effective_exposure_micros(self) -> int:
        """Exposure used for the invariant: the larger of our view and Kalshi's.

        Taking the maximum means a discrepancy always resolves in the safe
        direction. If we have somehow missed a fill, the exchange's number is
        higher and it is the one that binds.
        """
        return max(self.total_exposure_micros(), self.exchange_total_notional_micros)

    def effective_leverage(self) -> Decimal | None:
        if not self.equity_is_known or self.equity_micros <= 0:
            return None
        used = self.effective_exposure_micros() + self.total_pending_exposure_micros()
        return (Decimal(used) / Decimal(self.equity_micros)).quantize(Decimal("0.000001"))

    # -- construction --------------------------------------------------------

    @classmethod
    def from_exchange(
        cls,
        balance: PerpBalance,
        risk: PerpRiskSnapshot | None,
        *,
        subaccount: int = 0,
        updated_at_ms: int = 0,
    ) -> PerpAccountState:
        """Build state from exchange responses, computing equity conservatively."""
        state = cls(updated_at_ms=updated_at_ms)

        sub = balance.subaccount(subaccount)
        if sub is not None and sub.equity_is_reported:
            state.equity_micros = sub.account_equity_micros
            state.equity_source = "margin_balance.account_equity"
            state.equity_is_known = True
        else:
            # Documented case: account_equity is 0 for self-clearing members.
            # Reconstruct it as cash plus mark-to-market position value.
            derived = balance.settled_funds_micros + balance.total_position_value_micros
            if derived > 0:
                state.equity_micros = derived
                state.equity_source = "settled_funds + position_value"
                state.equity_is_known = True
            else:
                state.equity_micros = 0
                state.equity_source = "unavailable"
                state.equity_is_known = False

        if risk is not None:
            state.exchange_total_notional_micros = risk.total_position_notional_micros
            state.exchange_account_leverage = risk.exchange_account_leverage
            state.exchange_maintenance_margin_micros = risk.total_maintenance_margin_micros
            for pos in risk.positions:
                if pos.position_qc == 0:
                    continue
                state.positions[pos.ticker] = PerpPositionState(
                    ticker=pos.ticker,
                    position_qc=pos.position_qc,
                    mark_price_cc=pos.mark_price_cc,
                )
        return state


class PerpRiskEngine:
    """Independent risk engine for perpetual futures.

    Shares no calculation with the event-contract engine. Perpetuals have
    margin, funding, marks and liquidation; event contracts have none of those,
    and a single generic engine would have to ignore all of them.
    """

    def __init__(self, settings: PerpRiskSettings, *, max_state_age_ms: int = 30_000) -> None:
        self.settings = settings
        self.max_state_age_ms = max_state_age_ms
        # Verify the invariant is intact at construction, so a compromised build
        # fails at start-up rather than at the first order.
        assert_invariant_intact()

    # -- the main entry point ------------------------------------------------

    def evaluate(
        self,
        intent: OrderIntent,
        state: PerpAccountState,
        *,
        book: OrderBook | None = None,
        now_ms: int,
        estimated_fees_micros: int = 0,
        funding_rate_bps_per_hour: Decimal | None = None,
    ) -> RiskVerdict:
        """Decide whether a perpetual order may be sent, and at what size.

        Checks run in order of severity: anything that makes the decision
        *unsafe to make at all* (wrong domain, stale state, unknown mark) runs
        before anything that merely sizes the order down.
        """
        if intent.market_type is not MarketType.PERP:
            return RiskVerdict.reject(
                intent, "WRONG_DOMAIN",
                f"perpetual risk engine received a {intent.market_type.value} intent",
            )

        checks: list[CheckResult] = []

        # --- 1. state must be fresh and complete ----------------------------
        age_ms = now_ms - state.updated_at_ms if state.updated_at_ms else None
        if age_ms is None or age_ms > self.max_state_age_ms:
            checks.append(
                CheckResult(
                    "account_state_freshness", False,
                    limit=f"<= {self.max_state_age_ms}ms",
                    observed=f"{age_ms}ms" if age_ms is not None else "never updated",
                )
            )
            return self._rejected(
                intent, "STALE_ACCOUNT_STATE",
                "perpetual account state is stale; refusing to compute leverage from it",
                checks,
            )
        checks.append(
            CheckResult("account_state_freshness", True, limit=f"<= {self.max_state_age_ms}ms",
                        observed=f"{age_ms}ms")
        )

        position = state.positions.get(intent.ticker) or PerpPositionState(ticker=intent.ticker)

        # A mark price is required: exposure is |qty| * mark, and without a mark
        # there is no exposure figure and therefore no leverage check.
        mark_cc = position.mark_price_cc
        if mark_cc is None and book is not None:
            mark_cc = book.mid_cc
        if mark_cc is None or mark_cc <= 0:
            checks.append(CheckResult("mark_price_available", False, observed="none"))
            return self._rejected(
                intent, "NO_MARK_PRICE",
                "no usable mark price; perpetual exposure cannot be computed",
                checks,
            )
        checks.append(CheckResult("mark_price_available", True, observed=str(mark_cc)))
        position = PerpPositionState(
            ticker=position.ticker,
            position_qc=position.position_qc,
            mark_price_cc=mark_cc,
            entry_price_cc=position.entry_price_cc,
            contract_multiplier_micros=position.contract_multiplier_micros,
            pending_buy_qc=position.pending_buy_qc,
            pending_sell_qc=position.pending_sell_qc,
        )

        # --- 2. leverage: the hard invariant --------------------------------
        # Determine the *signed* quantity this order would add.
        signed_qty = intent.qty_qc if intent.book_side is BookSide.BID else -intent.qty_qc

        assessment = self._assess_leverage(intent, state, position, signed_qty)
        checks.append(
            CheckResult(
                "perp_leverage_1x", assessment.permitted,
                limit="1.00x",
                observed=(
                    f"{assessment.projected_leverage:.6f}x"
                    if assessment.projected_leverage is not None
                    else "unknown"
                ),
                detail=assessment.reason,
            )
        )
        if not assessment.permitted:
            metrics.increment(M_LEVERAGE_BLOCKS)
            log.error(
                "risk.leverage_blocked",
                ticker=intent.ticker,
                strategy_id=intent.strategy_id,
                reason_code=assessment.reason_code,
                projected_leverage=(
                    str(assessment.projected_leverage)
                    if assessment.projected_leverage is not None else None
                ),
                equity_usd=str(micros_to_decimal(assessment.equity_micros)),
                projected_exposure_usd=str(
                    micros_to_decimal(assessment.projected_exposure_micros)
                ),
            )
            return self._rejected(
                intent, assessment.reason_code, assessment.reason, checks,
                leverage=assessment.to_dict(),
            )

        # --- 3. size the order within the remaining limits ------------------
        approved_qty = intent.qty_qc

        # An order is "reducing" when it moves the position toward flat. The
        # sizing caps below exist to limit how much exposure may be *taken on*,
        # so none of them apply to an order that only sheds exposure -- and in
        # particular they must not block a flatten when equity is unknown, which
        # is exactly when flattening matters most.
        is_reducing = intent.reduce_only or (signed_qty * position.position_qc < 0)

        # 3a. Configured exposure budget (can only be tighter than 1.00x).
        approved_qty = self._apply_exposure_budget(
            approved_qty, intent, state, position, signed_qty, mark_cc, checks,
            is_reducing=is_reducing,
        )
        if approved_qty <= 0:
            return self._rejected(
                intent, "EXPOSURE_BUDGET_EXHAUSTED",
                "no perpetual exposure budget remains under the configured limit",
                checks, leverage=assessment.to_dict(),
            )

        # 3b. Per-position cap.
        approved_qty = self._apply_position_cap(
            approved_qty, state, position, signed_qty, mark_cc, checks,
            is_reducing=is_reducing,
        )
        if approved_qty <= 0:
            return self._rejected(
                intent, "POSITION_LIMIT",
                "per-position perpetual limit leaves no room for this order",
                checks, leverage=assessment.to_dict(),
            )

        # 3c. Absolute per-order notional cap.
        max_order_micros = int(self.settings.max_order_notional_usd * 1_000_000)
        order_notional = approved_qty * intent.limit_price_cc
        if order_notional > max_order_micros:
            approved_qty = max_order_micros // intent.limit_price_cc if intent.limit_price_cc else 0
            checks.append(
                CheckResult(
                    "max_order_notional", approved_qty > 0,
                    limit=f"${self.settings.max_order_notional_usd}",
                    observed=str(micros_to_decimal(order_notional)),
                    detail="order resized down to the per-order notional cap",
                )
            )
            if approved_qty <= 0:
                return self._rejected(
                    intent, "ORDER_TOO_LARGE",
                    "order exceeds the maximum permitted notional even at one contract",
                    checks, leverage=assessment.to_dict(),
                )
        else:
            checks.append(
                CheckResult("max_order_notional", True,
                            limit=f"${self.settings.max_order_notional_usd}",
                            observed=str(micros_to_decimal(order_notional)))
            )

        # 3d. Open-position count.
        if not intent.reduce_only and intent.ticker not in state.positions:
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
                    f"already holding {open_count} perpetual positions",
                    checks, leverage=assessment.to_dict(),
                )

        # --- 4. market quality ----------------------------------------------
        slippage_cc = None
        if book is not None:
            quality_ok, slippage_cc = self._check_market_quality(book, approved_qty, intent, checks)
            if not quality_ok:
                return self._rejected(
                    intent, "MARKET_QUALITY",
                    "market is too wide, too thin, or would slip too much",
                    checks, leverage=assessment.to_dict(),
                )

        # --- 5. funding cost -------------------------------------------------
        if funding_rate_bps_per_hour is not None:
            ok = abs(funding_rate_bps_per_hour) <= self.settings.max_funding_rate_bps_per_hour
            checks.append(
                CheckResult("funding_rate", ok,
                            limit=f"{self.settings.max_funding_rate_bps_per_hour} bps/hr",
                            observed=f"{funding_rate_bps_per_hour} bps/hr")
            )
            if not ok and not intent.reduce_only:
                return self._rejected(
                    intent, "FUNDING_TOO_EXPENSIVE",
                    "funding rate exceeds what the expected edge can support",
                    checks, leverage=assessment.to_dict(),
                )

        # --- 6. exchange notional risk limit ---------------------------------
        if state.exchange_notional_risk_limit_micros is not None:
            projected = state.effective_exposure_micros() + approved_qty * mark_cc
            ok = projected <= state.exchange_notional_risk_limit_micros
            checks.append(
                CheckResult("exchange_notional_risk_limit", ok,
                            limit=str(micros_to_decimal(state.exchange_notional_risk_limit_micros)),
                            observed=str(micros_to_decimal(projected)))
            )
            if not ok:
                return self._rejected(
                    intent, "EXCHANGE_NOTIONAL_LIMIT",
                    "order would exceed the notional risk limit Kalshi has set for this account",
                    checks, leverage=assessment.to_dict(),
                )

        # --- 7. re-run the invariant at the FINAL approved size --------------
        # Resizing above could in principle have been computed wrongly; the
        # invariant is therefore checked again against exactly what will be sent.
        final_signed = approved_qty if intent.book_side is BookSide.BID else -approved_qty
        final_assessment = self._assess_leverage(intent, state, position, final_signed)
        if not final_assessment.permitted:
            metrics.increment(M_LEVERAGE_BLOCKS)
            log.error(
                "risk.leverage_blocked_on_final_size",
                ticker=intent.ticker, approved_qty_qc=approved_qty,
                reason_code=final_assessment.reason_code,
            )
            return self._rejected(
                intent, final_assessment.reason_code, final_assessment.reason, checks,
                leverage=final_assessment.to_dict(),
            )

        decision = (
            RiskDecision.RESIZED if approved_qty != intent.qty_qc else RiskDecision.APPROVED
        )
        return RiskVerdict(
            decision=decision,
            intent=intent,
            approved_qty_qc=approved_qty,
            reason_code="APPROVED",
            reason=(
                "within all perpetual risk limits"
                if decision is RiskDecision.APPROVED
                else "approved at a reduced size to stay within perpetual risk limits"
            ),
            checks=checks,
            leverage=final_assessment.to_dict(),
            # Restated for the size actually approved; see the note in
            # EventRiskEngine.evaluate. A resize here is the common case, because
            # the leverage invariant caps size far more often than it refuses.
            estimated_fees_micros=prorate_fee_micros(
                estimated_fees_micros,
                from_qty_qc=intent.qty_qc,
                to_qty_qc=approved_qty,
            ),
            expected_slippage_cc=slippage_cc,
            current_exposure_micros=state.effective_exposure_micros(),
            projected_exposure_micros=final_assessment.projected_exposure_micros,
        )

    # -- leverage ------------------------------------------------------------

    def _assess_leverage(
        self,
        intent: OrderIntent,
        state: PerpAccountState,
        position: PerpPositionState,
        signed_qty: int,
    ) -> LeverageAssessment:
        """Compute the exposure delta and hand it to the invariant.

        The delta is computed *per market from signed positions*, so a sell
        against an existing long correctly reduces exposure rather than adding
        to it. That is what makes reduce-only and flatten orders work without
        any exemption from the rule.
        """
        current_total = state.effective_exposure_micros()
        pending_total = state.total_pending_exposure_micros()

        before = position.exposure_micros()
        after = position.projected_exposure_micros(signed_qty)
        delta = after - before

        if not state.equity_is_known:
            # Force the invariant down its unknown-equity path rather than
            # passing a fabricated number.
            return evaluate_perp_leverage(
                equity_micros=0,
                current_exposure_micros=current_total,
                pending_exposure_micros=pending_total,
                requested_delta_micros=delta,
            )

        return evaluate_perp_leverage(
            equity_micros=state.equity_micros,
            current_exposure_micros=current_total,
            pending_exposure_micros=pending_total,
            requested_delta_micros=delta,
        )

    # -- sizing helpers ------------------------------------------------------

    def _apply_exposure_budget(
        self,
        qty: int,
        intent: OrderIntent,
        state: PerpAccountState,
        position: PerpPositionState,
        signed_qty: int,
        mark_cc: int,
        checks: list[CheckResult],
        *,
        is_reducing: bool,
    ) -> int:
        """Apply the configured exposure fraction, which is always <= 1.00x."""
        if is_reducing:
            checks.append(CheckResult("exposure_budget", True, detail="exposure-reducing order"))
            return qty
        if not state.equity_is_known or state.equity_micros <= 0:
            checks.append(CheckResult("exposure_budget", False, observed="equity unknown"))
            return 0

        budget_micros = int(
            Decimal(state.equity_micros) * self.settings.max_exposure_pct_equity
        )
        used = state.effective_exposure_micros() + state.total_pending_exposure_micros()
        headroom = budget_micros - used
        if headroom <= 0:
            checks.append(
                CheckResult("exposure_budget", False,
                            limit=str(micros_to_decimal(budget_micros)),
                            observed=str(micros_to_decimal(used)))
            )
            return 0

        max_qty = headroom // mark_cc
        allowed = min(qty, max_qty)
        checks.append(
            CheckResult("exposure_budget", allowed > 0,
                        limit=str(micros_to_decimal(budget_micros)),
                        observed=str(micros_to_decimal(used)),
                        detail=None if allowed == qty else "resized to remaining budget")
        )
        return max(0, allowed)

    def _apply_position_cap(
        self,
        qty: int,
        state: PerpAccountState,
        position: PerpPositionState,
        signed_qty: int,
        mark_cc: int,
        checks: list[CheckResult],
        *,
        is_reducing: bool,
    ) -> int:
        if is_reducing:
            checks.append(CheckResult("max_position_size", True, detail="exposure-reducing order"))
            return qty
        if not state.equity_is_known or state.equity_micros <= 0:
            checks.append(CheckResult("max_position_size", False, observed="equity unknown"))
            return 0
        cap_micros = int(Decimal(state.equity_micros) * self.settings.max_position_pct_equity)
        # ``qty`` is always positive; apply it in the direction of the intent.
        direction = 1 if signed_qty >= 0 else -1
        projected_after = abs(position.position_qc + direction * qty)
        projected_micros = projected_after * mark_cc
        if projected_micros <= cap_micros:
            checks.append(
                CheckResult("max_position_size", True,
                            limit=str(micros_to_decimal(cap_micros)),
                            observed=str(micros_to_decimal(projected_micros)))
            )
            return qty

        max_total_qty = cap_micros // mark_cc
        allowed = max(0, max_total_qty - abs(position.position_qc))
        allowed = min(qty, allowed)
        checks.append(
            CheckResult("max_position_size", allowed > 0,
                        limit=str(micros_to_decimal(cap_micros)),
                        observed=str(micros_to_decimal(projected_micros)),
                        detail="resized to the per-position cap")
        )
        return allowed

    def _check_market_quality(
        self, book: OrderBook, qty: int, intent: OrderIntent, checks: list[CheckResult]
    ) -> tuple[bool, int | None]:
        spread = book.spread_cc
        spread_ok = spread is not None and spread <= self.settings.max_spread_cc
        checks.append(
            CheckResult("max_spread", spread_ok,
                        limit=str(self.settings.max_spread_cc),
                        observed=str(spread) if spread is not None else "one-sided book")
        )
        if not spread_ok and not intent.reduce_only:
            return False, None

        depth = book.total_depth_qc(is_bid=not intent.is_buy)
        depth_ok = depth >= self.settings.min_book_depth_qc
        checks.append(
            CheckResult("min_book_depth", depth_ok,
                        limit=str(self.settings.min_book_depth_qc), observed=str(depth))
        )
        if not depth_ok and not intent.reduce_only:
            return False, None

        slippage = book.slippage_cc(qty, is_buy=intent.is_buy)
        slip_ok = slippage is not None and slippage <= self.settings.max_slippage_cc
        checks.append(
            CheckResult("max_slippage", slip_ok,
                        limit=str(self.settings.max_slippage_cc),
                        observed=str(slippage) if slippage is not None else "insufficient depth")
        )
        if not slip_ok and not intent.reduce_only:
            return False, slippage
        return True, slippage

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _rejected(
        intent: OrderIntent,
        code: str,
        reason: str,
        checks: list[CheckResult],
        *,
        leverage: dict | None = None,
    ) -> RiskVerdict:
        metrics.increment(M_RISK_REJECTIONS, domain="perp", reason=code)
        return RiskVerdict.reject(intent, code, reason, checks=checks, leverage=leverage)

    def enforce(
        self,
        intent: OrderIntent,
        state: PerpAccountState,
        *,
        now_ms: int,
        **kwargs: object,
    ) -> RiskVerdict:
        """Evaluate and raise on rejection.

        Used by paths that treat a rejection as exceptional. Ordinary flow calls
        :meth:`evaluate` and inspects the verdict, because most intents are
        expected to be rejected and exceptions are the wrong tool for that.
        """
        verdict = self.evaluate(intent, state, now_ms=now_ms, **kwargs)  # type: ignore[arg-type]
        if not verdict.approved:
            if verdict.reason_code in ("LEVERAGE_WOULD_EXCEED_1X", "EQUITY_UNKNOWN_OR_NON_POSITIVE"):
                raise LeverageViolation(verdict.reason, detail=verdict.leverage or {})
            from kat.common.errors import RiskRejection

            raise RiskRejection(verdict.reason_code, verdict.reason, detail=verdict.to_dict())
        return verdict
