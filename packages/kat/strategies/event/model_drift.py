"""Trade the drift a forward-move model predicts.

The platform could train a model and had nothing that could use one. The only
model-consuming strategy, :mod:`~kat.strategies.event.mispricing`, needs the
probability that a contract *resolves YES*, and that target needs settled markets
the platform had never recorded. So the trainer produced the target it could
label -- whether the mid rises by more than a threshold over a horizon -- and
nothing consumed it. Fed to the mispricing strategy it was read as a fair value
and produced systematically wrong trades, which is why that strategy now refuses
it outright.

This is the consumer that target deserves.

From a probability to a price
-----------------------------
The model answers "will the mid rise by more than ``T`` centicents in the next
``H`` minutes". Converting that into a fair value needs one assumption, and the
conservative one is symmetry: treat a miss as an adverse move of the same size
the hit would have gained. Expected drift is then::

    drift = T * p - T * (1 - p) = T * (2p - 1)

At ``p = 0.5`` that is zero, which is the right answer for a model with no view.
It is deliberately pessimistic -- a miss is usually a small move rather than a
full adverse ``T`` -- because a fair value that overstates the edge is how a
strategy talks itself into paying the spread.

The threshold is read from the model's own target rather than configured here.
``forward_move_30m_200cc`` and ``forward_move_30m_50cc`` are different questions,
and a strategy that assumed one while served the other would misprice every trade
by the ratio between them. A target it cannot parse is a target it will not
trade.

Why the threshold matters economically
--------------------------------------
The label's own docstring asks for the threshold to be set to the round-trip
cost -- "will this move enough to pay for itself?" -- and the default of 50
centicents was used instead, against real round-trip costs of 136 to 550. So the
model was trained to predict moves too small to be worth capturing. Retrained at
200 centicents it scores 0.857 AUC out of sample against 0.80, and passes every
selection gate with a score of 0.82 against 0.66.
"""

from __future__ import annotations

import re

from kat.common.enums import MarketType, OutcomeSide
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import TradingStrategy, clamp_price_cc
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    LIQUIDITY_PARAM_SPACE,
    net_ev_floor_micros,
    propose_event_trade,
    screen_event_market,
    screen_freshness,
)

__all__ = ["ModelDriftStrategy"]

_SKIP_NO_MODEL = "no_model_probability"
_SKIP_WRONG_TARGET = "model_answers_a_different_question"
_SKIP_NO_VIEW = "model_has_no_directional_view"
_SKIP_EDGE_TOO_SMALL = "drift_below_threshold"
_SKIP_PRICE_BAND = "price_outside_band"

#: Targets this strategy can read: a forward move over a horizon, with the
#: threshold that defines "a move" written into the name.
_TARGET = re.compile(r"^forward_move_(\d+)m_(\d+)cc$")


class ModelDriftStrategy(TradingStrategy):
    """Buys the side a forward-move model expects the mid to drift toward."""

    FAMILY = "model_drift"
    NAME = "Model Drift"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.EVENT
    DESCRIPTION = """
    Reads a forward-move model as an expected drift rather than as a fair
    value, using the move threshold the model was trained on, and buys the
    side that drift points to only when the edge survives a round trip.
    """

    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "min_edge_cc", 150, low=25, high=1_000, step=25,
            description="drift must beat the touch by this much before it is worth crossing",
        ),
        StrategyParam(
            "min_confidence", 0.60, low=0.50, high=0.95, step=0.01,
            description="how far from a coin flip the model must be to have a view at all",
        ),
        StrategyParam(
            "min_price_cc", 300, low=100, high=5_000, step=100,
            description="cheapest midpoint worth trading; below it the payoff is a rounding error",
        ),
        StrategyParam(
            "max_price_cc", 9_700, low=5_000, high=9_900, step=100,
            description="dearest midpoint worth trading",
        ),
    )
    REQUIRES_FEATURES = ()
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def warmup_ms(self) -> int:
        """No history of its own: the model carries it."""
        return 0

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        probability = ctx.model_probability
        if probability is None or not 0.0 < probability < 1.0:
            return self.skip(_SKIP_NO_MODEL)

        threshold_cc = self._threshold_cc(ctx.model_target)
        if threshold_cc is None:
            return self.skip(_SKIP_WRONG_TARGET)

        # Distance from a coin flip, not the probability itself: a model saying
        # 0.5 has no view, and one saying 0.02 has a strong view in the other
        # direction.
        conviction = abs(probability - 0.5) * 2.0
        if conviction < (self.float_param("min_confidence") - 0.5) * 2.0:
            return self.skip(_SKIP_NO_VIEW)

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

        drift_cc = int(threshold_cc * (2.0 * probability - 1.0))
        fair_value_cc = clamp_price_cc(mid_cc + drift_cc)

        # Both sides against their own executable price, exactly as the
        # mispricing strategy does: the NO edge is (1 - fair) - no_ask.
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
            confidence=min(0.99, conviction),
            costing=proposal.costing,
            reason="model_drift",
            extra={
                "model_probability": round(probability, 6),
                "model_target": ctx.model_target,
                "model_id": ctx.model_id,
                "model_version": ctx.model_version,
                "threshold_cc": threshold_cc,
                "drift_cc": drift_cc,
                "market_mid_cc": mid_cc,
                "touch_edge_cc": edge_cc,
                "vwap_cc": proposal.execution.vwap_cc,
            },
        )
        return self.emit([intent])

    @staticmethod
    def _threshold_cc(target: str | None) -> int | None:
        """The move size the model was trained on, taken from its own target.

        Returns ``None`` for a target this strategy cannot read, which includes
        the settlement target and any forward-move target that does not name its
        threshold. Assuming a threshold would misprice every trade by the ratio
        between the assumed one and the real one.
        """
        if not target:
            return None
        match = _TARGET.match(target)
        if match is None:
            return None
        threshold = int(match.group(2))
        return threshold if 0 < threshold < CC_PER_DOLLAR else None
