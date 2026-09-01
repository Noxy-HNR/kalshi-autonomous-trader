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

    drift = T * (p - base) / (1 - base)   when p is above the base rate
    drift = T * (p - base) / base         when it is below

At ``p = base_rate`` that is zero, which is the right answer for a model saying
nothing beyond what is already known.

The base rate, not a half, is the neutral point. A half is neutral only for a
balanced target; this model's event happens 2.4% of the time, so every
probability it emits is below a half and reading it against 0.5 made ``2p - 1``
permanently near -1 -- a constant "sell" wearing the costume of a signal. It
produced 37 completed round trips in a week, every one short, and a win rate of
exactly zero.
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
from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT
from kat.kalshi.fees import realised_fee_micros
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    TradingStrategy,
    clamp_price_cc,
    event_costing,
    fee_schedule_for,
)
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
_SKIP_NO_BASE_RATE = "model_reports_no_base_rate"
_SKIP_WRONG_TARGET = "model_answers_a_different_question"
_SKIP_NO_VIEW = "model_has_no_directional_view"
_SKIP_EDGE_TOO_SMALL = "drift_below_threshold"
_SKIP_PRICE_BAND = "price_outside_band"
_SKIP_NEAR_BOUND = "too_close_to_a_bound_for_the_sign_to_mean_anything"
_SKIP_HOLDING = "holding_while_the_view_stands"

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
            "min_edge_cc", 25, low=25, high=1_000, step=25,
            description="floor under the cost test, for degenerate books rather than economics",
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

        # Judged against how often the event actually happens, not against a
        # coin flip.
        #
        # A half is the neutral point only for a balanced target. This model's
        # event happens 2.4% of the time, so *every* probability it emits is
        # below a half, and reading it against 0.5 makes 2p-1 permanently near
        # -1. The strategy was not weakly bearish, it was a constant: 37
        # completed round trips in a week, every one short, a win rate of
        # exactly zero, and the fees to re-enter that non-view every thirty
        # minutes.
        #
        # Above the base rate is bullish, below it bearish, each normalised by
        # the room available on that side so both directions can reach full
        # conviction.
        base_rate = ctx.model_base_rate
        if base_rate is None or not 0.0 < base_rate < 1.0:
            return self.skip(_SKIP_NO_BASE_RATE)
        if probability >= base_rate:
            signal = (probability - base_rate) / (1.0 - base_rate)
        else:
            signal = (probability - base_rate) / base_rate
        conviction = abs(signal)
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

        # Near a bound the probability is not a directional view.
        #
        # The model answers "will the mid rise by T". A contract at 96 cents is
        # four cents from the ceiling, so it can barely rise by 200 centicents
        # whatever happens next, and the model correctly reports a probability
        # near zero. Read against a symmetric 0.5 prior that becomes maximum
        # bearish conviction -- and it is nothing of the kind, it is arithmetic
        # about where the price sits.
        #
        # Measured: without this the strategy shorted the same two 96-cent
        # markets every thirty minutes for a week. 37 completed round trips, a
        # win rate of exactly zero, -7.2% and a Sharpe of -47, with more than
        # half the loss in fees paid to re-enter a view that was never a view.
        #
        # Comparing p to 0.5 is only meaningful where a move of T is genuinely
        # available in both directions with room to spare.
        headroom_cc = 2 * threshold_cc
        if not headroom_cc <= mid_cc <= CC_PER_DOLLAR - headroom_cc:
            return self.skip(_SKIP_NEAR_BOUND)

        drift_cc = int(threshold_cc * signal)
        fair_value_cc = clamp_price_cc(mid_cc + drift_cc)

        # Exits come first, and they come before the cost test.
        #
        # A drift strategy with no exit is not a strategy, it is an accumulator.
        # Replayed over a week it opened 36 positions, closed none of them, and
        # finished down 0.59% -- of which $1.51 was fees on a $200 account. Every
        # position was still open at the end, so nothing it predicted was ever
        # realised; it paid to enter, held, and paid again in marks.
        #
        # The view that opened a position is the view that should close it. When
        # the model's drift turns against what is held, the reason for holding is
        # gone, and waiting for a fresh entry-quality edge in the other direction
        # is waiting for a much rarer event than the one that already happened.
        exit_intent = self._exit_if_reversed(ctx, drift_cc, quote)
        if exit_intent is not None:
            return self.emit([exit_intent])

        # Both sides against their own executable price, exactly as the
        # mispricing strategy does: the NO edge is (1 - fair) - no_ask.
        yes_edge_cc = fair_value_cc - quote.yes_ask_cc
        no_edge_cc = (CC_PER_DOLLAR - fair_value_cc) - quote.no_ask_cc
        side = OutcomeSide.YES if yes_edge_cc >= no_edge_cc else OutcomeSide.NO
        edge_cc = max(yes_edge_cc, no_edge_cc)

        # The edge has to beat what capturing it costs, and that cost is a
        # function of price rather than a constant. The fee is quadratic --
        # 175 centicents a side at $0.50, 33 at $0.95, 14 at $0.98 -- so a
        # round trip runs about 450 centicents mid-board and about 130 at the
        # edges, against a drift this model can never push past its own 200.
        #
        # A fixed floor gets this wrong in both directions: at mid prices it
        # lets through trades the fees will eat, and at the extremes it blocks
        # trades that comfortably pay. Requiring the edge to cover the actual
        # round trip is the same test at every price, and it is what decides
        # where this model can be traded at all rather than a number chosen to
        # produce trades.
        # The cost is the bar. min_edge_cc is only a backstop against a
        # degenerate book quoting no spread at all, and is deliberately far
        # below the cost at any real price -- a constant set above it silently
        # replaces the economics with a number somebody picked. At 150 it did
        # exactly that: a 5-cent market with a 1-cent spread costs 116
        # centicents to get out of and the model offered 142, which pays, and
        # the constant refused it.
        cost_cc = self._round_trip_cc(mid_cc, quote.spread_cc)  # exit + fees
        floor_cc = max(self.int_param("min_edge_cc"), cost_cc)
        if edge_cc < floor_cc:
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

    def _exit_if_reversed(
        self, ctx: StrategyContext, drift_cc: int, quote: object
    ) -> OrderIntent | None:
        """Flatten a position the model no longer supports.

        Returns ``None`` when there is nothing held or the view still stands.

        Not subject to the cost floor that governs entries. That floor asks
        whether an edge is worth paying to capture; this is asking whether to
        keep paying to hold a position whose reason has gone. Kalshi is exited by
        buying the other outcome, so this is expressed as a reduce-only buy of
        the opposite side and the risk engine verifies it genuinely reduces.
        """
        position_qc = ctx.position_qc
        if position_qc == 0:
            return None

        holding_yes = position_qc > 0
        drift_up = drift_cc > 0
        reversed_view = holding_yes != drift_up

        # A forecast expires. The model was asked what happens over one horizon,
        # so once that horizon has passed the position is being held on nothing
        # -- and on this data the view almost never reverses, because the
        # probability sits stably near its base rate. Without the clock the
        # reversal test alone closed nothing at all.
        horizon_ms = self._horizon_ms(ctx.model_target)
        age_ms = ctx.position_age_ms
        expired = (
            horizon_ms is not None and age_ms is not None and age_ms >= horizon_ms
        )

        if not reversed_view and not expired:
            return None

        side = OutcomeSide.NO if holding_yes else OutcomeSide.YES
        price_cc = quote.no_ask_cc if holding_yes else quote.yes_ask_cc  # type: ignore[attr-defined]
        if not price_cc:
            return None

        intent = self.event_intent(
            ctx,
            side=side,
            qty_qc=abs(position_qc),
            limit_price_cc=price_cc,
            fair_value_cc=clamp_price_cc(quote.mid_cc + drift_cc),  # type: ignore[attr-defined]
            confidence=min(0.99, abs(drift_cc) / max(1, abs(drift_cc))),
            costing=event_costing(
                side=side,
                cost_cc=price_cc,
                fair_value_cc=clamp_price_cc(quote.mid_cc + drift_cc),  # type: ignore[attr-defined]
                qty_qc=abs(position_qc),
                schedule=fee_schedule_for(ctx),
            ),
            reason="model_drift_exit",
            extra={
                "model_probability": round(ctx.model_probability or 0.0, 6),
                "model_target": ctx.model_target,
                "held_qc": position_qc,
                "drift_cc": drift_cc,
                "held_ms": age_ms,
                "closed_because": "reversed" if reversed_view else "horizon_expired",
            },
        )
        intent.reduce_only = True
        return intent

    @staticmethod
    def _round_trip_cc(mid_cc: int, spread_cc: int) -> int:
        """What is left to pay *after* the edge has been measured, in centicents.

        Not the whole round trip. The edge is ``fair - ask``, and the ask is
        already half a spread above the mid, so the entry half-spread has been
        deducted before this floor is applied. Charging the full spread here
        counts it twice and rejects trades by half a spread more than the venue
        actually asks for.

        What remains is the exit half-spread and both fees. The realised fee
        rather than the padded estimate: this is asking what the venue charges,
        not deciding a trade under uncertainty.
        """
        fee_micros = realised_fee_micros(price_cc=mid_cc, qty_qc=QC_PER_CONTRACT)
        # micros to centicents: 1e-6 USD against 1e-4 USD.
        return spread_cc // 2 + 2 * (fee_micros // 100)

    @staticmethod
    def _horizon_ms(target: str | None) -> int | None:
        """How far ahead the model was asked to see, from its own target.

        A forecast is only actionable for as long as it forecasts. Holding a
        30-minute view for a week is not the strategy the model was trained for,
        and it is what turned this one into an accumulator: replayed over a week
        it opened 36 positions, closed none, and finished down 0.59% -- almost
        all of it fees and marks on trades whose forecast had long since expired.
        """
        if not target:
            return None
        match = _TARGET.match(target)
        return int(match.group(1)) * 60_000 if match else None

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
