"""Trade a model probability against the quoted price, net of what it costs.

This is the most direct strategy in the platform and the easiest one to fool
yourself with: a model says 62%, the market says 58 cents, and it looks like
four cents of free money. It is not, for three reasons that this module treats
as first-class rather than as caveats.

**The market is a forecaster too, and usually a better one.** The model's number
is shrunk toward the market midpoint before it is used. The weight on the market
is a parameter, and it is raised further when the model reports low confidence,
so an unreliable model converges to "the market is right" and produces no
trades. That is the correct failure mode for a model that has stopped working.

**The edge is measured against what we would pay, not against the midpoint.**
Buying YES costs the ask and buying NO costs one dollar minus the bid; a
four-cent edge to the midpoint on a three-cent market is a two-and-a-half-cent
edge in reality, before fees.

**Fees are charged round trip, at the size actually traded.** Kalshi's quadratic
fee is largest at fifty cents, which is exactly where a probability model has
the least to say. :func:`~kat.strategies.proposal.propose_event_trade` charges
entry and exit and refuses anything whose net expected value falls below the
configured floor.

The extreme-price band is a fourth guard with a different motivation. Below a
few cents the fee is a large fraction of the price and the model's calibration
is worst; above ninety-odd cents the payoff is a few cents against a
near-certain dollar of capital at risk. Both tails are excluded outright rather
than being left for the expected-value arithmetic to discover, because in both
of them the arithmetic is dominated by a probability estimate we do not trust.
"""

from __future__ import annotations

from kat.common.enums import MarketType, OutcomeSide
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    F_MODEL_CONFIDENCE,
    TradingStrategy,
    conservative_fair_value_cc,
)
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    LIQUIDITY_PARAM_SPACE,
    net_ev_floor_micros,
    propose_event_trade,
    screen_event_market,
    screen_freshness,
)

__all__ = ["MispricingStrategy"]

_SKIP_NO_MODEL = "no_model_probability"
_SKIP_PRICE_BAND = "price_outside_band"
_SKIP_EDGE_TOO_SMALL = "edge_below_threshold"


class MispricingStrategy(TradingStrategy):
    """Buy the side of an event contract the model says is underpriced."""

    FAMILY = "mispricing"
    NAME = "Model Mispricing"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.EVENT
    DESCRIPTION = """
    Compares a deployed model's probability with the price actually available
    on each side of the book, shrinks the model toward the market in proportion
    to its own reported unreliability, and buys only when the remaining edge
    survives a round trip of fees at the size the book can fill.
    """
    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "min_edge_cc", 150, low=25, high=2_000, step=25,
            description="minimum edge over the executable price before costs are even computed",
        ),
        StrategyParam(
            "market_weight_bps", 3_000, low=0, high=9_000, step=250,
            description="baseline weight on the market midpoint when shrinking the model, in bps",
        ),
        StrategyParam(
            "assumed_confidence", 0.5, low=0.0, high=1.0, step=0.05,
            description="model confidence assumed when the model does not report one",
        ),
        StrategyParam(
            "min_price_cc", 500, low=100, high=4_000, step=100,
            description="cheapest midpoint this strategy will trade; below it fees dominate",
        ),
        StrategyParam(
            "max_price_cc", 9_500, low=6_000, high=9_900, step=100,
            description="dearest midpoint this strategy will trade; above it the payoff is a few cents",
        ),
    )
    # ``model_probability`` arrives on the context rather than through the
    # feature bus, so nothing here is required of the feature engine. Model
    # confidence is used when published and assumed conservatively when not,
    # which is why it is not declared as a requirement: demanding it would stop
    # the strategy running against a model that simply does not report one.
    REQUIRES_FEATURES = ()
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def warmup_ms(self) -> int:
        """No history of its own: the model carries it. Honestly zero."""
        return 0

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        probability = ctx.model_probability
        if probability is None or not 0.0 < probability < 1.0:
            return self.skip(_SKIP_NO_MODEL)

        quote, reason = screen_event_market(
            ctx,
            max_spread_cc=self.int_param("max_spread_cc"),
            min_depth_qc=self.int_param("min_depth_qc"),
        )
        if quote is None or reason is not None:
            return self.skip(reason or "no_quote")

        mid_cc = quote.mid_cc
        if not self.int_param("min_price_cc") <= mid_cc <= self.int_param("max_price_cc"):
            return self.skip(_SKIP_PRICE_BAND)

        fair_value_cc = self._shrunk_fair_value_cc(ctx, probability, mid_cc)

        # Both sides are evaluated against their own executable price. The NO
        # edge is (1 - fair) - no_ask, which reduces to yes_bid - fair.
        yes_edge_cc = fair_value_cc - quote.yes_ask_cc
        no_edge_cc = (CC_PER_DOLLAR - fair_value_cc) - quote.no_ask_cc
        side = OutcomeSide.YES if yes_edge_cc >= no_edge_cc else OutcomeSide.NO
        edge_cc = max(yes_edge_cc, no_edge_cc)
        if edge_cc < self.int_param("min_edge_cc"):
            return self.skip(_SKIP_EDGE_TOO_SMALL)

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
            confidence=self._confidence(ctx, proposal.edge_cc),
            costing=proposal.costing,
            reason="model_mispricing",
            extra={
                "model_probability": round(probability, 6),
                "model_id": ctx.model_id,
                "model_version": ctx.model_version,
                "market_mid_cc": mid_cc,
                "touch_edge_cc": edge_cc,
                "vwap_cc": proposal.execution.vwap_cc,
            },
        )
        return self.emit([intent])

    # -- internals -----------------------------------------------------------

    def _model_confidence(self, ctx: StrategyContext) -> float:
        raw = ctx.features.get(F_MODEL_CONFIDENCE)
        if raw is None:
            return self.float_param("assumed_confidence")
        return min(1.0, max(0.0, float(raw)))

    def _shrunk_fair_value_cc(self, ctx: StrategyContext, probability: float, mid_cc: int) -> int:
        """Blend the model toward the market, harder when the model is unsure.

        At zero confidence the fair value *is* the midpoint, so the edge is
        exactly zero and no trade can be constructed. That is deliberate: a
        model that does not believe itself should not move capital.
        """
        baseline = self.int_param("market_weight_bps") / 10_000.0
        confidence = self._model_confidence(ctx)
        market_weight = baseline + (1.0 - baseline) * (1.0 - confidence)
        blended_cc = (1.0 - market_weight) * (probability * CC_PER_DOLLAR) + market_weight * mid_cc
        return conservative_fair_value_cc(blended_cc, mid_cc)

    def _confidence(self, ctx: StrategyContext, edge_cc: int) -> float:
        """Conviction in [0, 1]: needs both a confident model and a real edge."""
        reference_cc = max(1, 2 * self.int_param("min_edge_cc"))
        strength = min(1.0, max(0.0, edge_cc / reference_cc))
        return min(1.0, max(0.0, self._model_confidence(ctx) * strength))
