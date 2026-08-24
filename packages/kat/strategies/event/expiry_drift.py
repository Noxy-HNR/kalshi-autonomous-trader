"""Buy the near-certain favourite shortly before resolution.

What this strategy claims, precisely
------------------------------------
As a market approaches resolution its price converges to zero or one. That fact
alone is worth nothing: if the price is a fair probability then the expected
value of holding to settlement is exactly the price, and convergence is not an
edge, it is arithmetic. Any strategy that says otherwise has assumed its
conclusion.

The claim made here is narrower and is a **prior, not a measurement**: prices in
the far tails of a binary market are biased, with favourites trading slightly
below their true probability and longshots slightly above -- the favourite/
longshot bias, which is one of the most replicated effects in binary betting
markets and which several published studies find in Kalshi-style contracts too.
This module encodes that prior as a single parameter, ``favourite_bias_cc``,
defaulting to one cent, and buys the favourite when the whole cent survives a
round trip of fees.

Why that is stated so loudly: ``favourite_bias_cc`` is in the search space with
a lower bound of zero, so a walk-forward test that does not find the bias in
this venue's data will drive it to zero and this strategy will stop trading. It
is designed to be falsifiable rather than to be right.

The gates that make it survivable
---------------------------------
* **A time window, at both ends.** Too early and the "near-certain" price is not
  yet near-certain. Too late and there is no counterparty to exit against, which
  turns a one-cent edge into a position held to settlement whether or not that
  was the plan.
* **A quiet tape.** The prior describes an undisturbed market. A market that has
  just moved several cents is being repriced by information, and a statistical
  bias measured across thousands of settled markets says nothing about this one
  in the middle of that.
* **Liquidity on both sides**, screened as everywhere else, because the position
  must be exitable.

The one thing this strategy must never become is "sell tail risk for a cent".
The price band has an upper bound as well as a lower one for exactly that
reason: above it the payoff is a fraction of a cent against a near-certain
dollar of committed capital, and the expected-value arithmetic stops being the
thing that protects you.
"""

from __future__ import annotations

from kat.common.enums import MarketType, OutcomeSide
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    F_MID_CHANGE_5M_CC,
    F_TIME_TO_EXPIRY_MS,
    TradingStrategy,
    conservative_fair_value_cc,
    time_to_expiry_ms,
)
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    LIQUIDITY_PARAM_SPACE,
    net_ev_floor_micros,
    propose_event_trade,
    screen_event_market,
    screen_freshness,
)

__all__ = ["ExpiryDriftStrategy"]

_SKIP_NO_EXPIRY = "time_to_expiry_unknown"
_SKIP_TOO_EARLY = "too_far_from_expiry"
_SKIP_TOO_LATE = "too_close_to_expiry"
_SKIP_NOT_FAVOURITE = "no_side_is_a_favourite"
_SKIP_UNSETTLED = "market_repricing_on_news"
_SKIP_NO_BIAS = "bias_parameter_is_zero"

#: The five-minute move feature is the only history required.
_WARMUP_MS = 300_000


class ExpiryDriftStrategy(TradingStrategy):
    """Buy the favourite in a quiet market inside its final hours."""

    FAMILY = "expiry_drift"
    NAME = "Expiry Favourite Drift"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.EVENT
    DESCRIPTION = """
    Encodes the favourite/longshot bias as a single falsifiable parameter and
    buys the favourite inside a bounded window before resolution, only while the
    market is quiet and only when the assumed bias survives a round trip of fees.
    """
    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "favourite_bias_cc", 100, low=0, high=500, step=25,
            description="assumed underpricing of the favourite, in centicents; zero disables trading",
        ),
        StrategyParam(
            "favourite_min_cc", 8_000, low=6_000, high=9_500, step=100,
            description="how strong the favourite must be before the prior is applied",
        ),
        StrategyParam(
            "favourite_max_cc", 9_600, low=8_000, high=9_900, step=50,
            description="beyond this the payoff is too small against the capital committed",
        ),
        StrategyParam(
            "min_ttl_ms", 900_000, low=60_000, high=7_200_000, step=60_000,
            description="stop trading this close to resolution; an exit needs a counterparty",
        ),
        StrategyParam(
            "max_ttl_ms", 21_600_000, low=3_600_000, high=604_800_000, step=3_600_000,
            description="do not start this far out; the price is not yet near-certain",
        ),
        StrategyParam(
            "max_recent_move_cc", 100, low=10, high=1_000, step=10,
            description="a larger five-minute move means the market is repricing on news",
        ),
    )
    REQUIRES_FEATURES = (F_TIME_TO_EXPIRY_MS, F_MID_CHANGE_5M_CC)
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def warmup_ms(self) -> int:
        return _WARMUP_MS

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        bias_cc = self.int_param("favourite_bias_cc")
        if bias_cc <= 0:
            # The optimiser is allowed to switch this strategy off entirely, and
            # when it has, saying so is more useful than an empty list.
            return self.skip(_SKIP_NO_BIAS)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        ttl_ms = time_to_expiry_ms(ctx)
        if ttl_ms is None:
            return self.skip(_SKIP_NO_EXPIRY)
        if ttl_ms > self.int_param("max_ttl_ms"):
            return self.skip(_SKIP_TOO_EARLY)
        if ttl_ms < self.int_param("min_ttl_ms"):
            return self.skip(_SKIP_TOO_LATE)

        recent_move_cc = abs(ctx.feature(F_MID_CHANGE_5M_CC))
        if recent_move_cc > self.int_param("max_recent_move_cc"):
            return self.skip(_SKIP_UNSETTLED)

        quote, reason = screen_event_market(
            ctx,
            max_spread_cc=self.int_param("max_spread_cc"),
            min_depth_qc=self.int_param("min_depth_qc"),
        )
        if quote is None or reason is not None:
            return self.skip(reason or "no_quote")

        mid_cc = quote.mid_cc
        favourite_cc = max(mid_cc, CC_PER_DOLLAR - mid_cc)
        if not (
            self.int_param("favourite_min_cc") <= favourite_cc <= self.int_param("favourite_max_cc")
        ):
            return self.skip(_SKIP_NOT_FAVOURITE)

        side = OutcomeSide.YES if mid_cc >= CC_PER_DOLLAR - mid_cc else OutcomeSide.NO
        # The bias belongs to the favourite. Expressed YES-denominated it lifts
        # fair value when YES is the favourite and lowers it when NO is.
        signed_bias_cc = bias_cc if side is OutcomeSide.YES else -bias_cc
        fair_value_cc = conservative_fair_value_cc(float(mid_cc + signed_bias_cc), mid_cc)

        book = ctx.market.book
        if book is None:
            return self.skip("no_usable_book")

        proposal, reason = propose_event_trade(
            self,
            ctx,
            book=book,
            quote=quote,
            side=side,
            fair_value_cc=fair_value_cc,
            max_position_qc=self.int_param("max_position_qc"),
            min_net_ev_micros=net_ev_floor_micros(self.float_param("min_net_ev_usd")),
        )
        if proposal is None:
            return self.skip(reason or "no_proposal")

        intent = self.event_intent(
            ctx,
            side=proposal.side,
            qty_qc=proposal.qty_qc,
            limit_price_cc=proposal.limit_price_cc,
            fair_value_cc=proposal.fair_value_cc,
            confidence=self._confidence(recent_move_cc),
            costing=proposal.costing,
            reason="expiry_favourite_drift",
            extra={
                "time_to_expiry_ms": ttl_ms,
                "favourite_price_cc": favourite_cc,
                "assumed_bias_cc": bias_cc,
                "recent_move_cc": round(recent_move_cc, 2),
                "market_mid_cc": mid_cc,
                # Recorded so the decision log shows this was a prior, not a
                # model output, when the trade is reviewed after the fact.
                "evidence": "prior",
            },
        )
        return self.emit([intent])

    # -- internals -----------------------------------------------------------

    def _confidence(self, recent_move_cc: float) -> float:
        """Capped at one half: this is a prior and never claims to be more.

        Within that ceiling, conviction falls as the tape gets busier, because
        the prior describes an undisturbed market.
        """
        budget_cc = max(1.0, float(self.int_param("max_recent_move_cc")))
        quiet = 1.0 - min(1.0, recent_move_cc / budget_cc)
        return 0.5 * quiet
