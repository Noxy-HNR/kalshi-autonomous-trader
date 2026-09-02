"""Rest the entry on a forecast, cross the exit, and see if the spread survives it.

This is one specific experiment, and it exists because the arithmetic on Kalshi
perpetuals says it is the only remaining place an edge could be. The numbers it
came from, measured out-of-sample over four days after a ten-day fit:

    median |15-minute move|              23.54 bp
    median round-trip cost (taker)       14.21 bp
    break-even accuracy, crossing        72.6%
    break-even accuracy, resting entry   63.5%
    model accuracy, top decile of
      confidence and volatility          65.1%

Crossing both legs, the model misses by seven points. Resting the entry removes
the spread from the cost -- the largest term in it -- and the bar drops below
what the model reaches. That gap is the entire hypothesis.

Why it might still fail, which is the point of running it
---------------------------------------------------------
The passive bar assumes a resting entry fills whenever the model fires. It does
not. A resting bid is reached when the market comes *back* to it, which is
disproportionately when the forecast was wrong or slow; it is missed when the
price runs away, which is when the forecast was right. Passive entry therefore
selects against exactly the trades worth having, and the margin here is 1.6
points against a bias whose size is unknown.

Nothing in a spreadsheet settles that. :mod:`kat.simulation.fills` assumes we
sit behind all the size already resting at our price and fills only when the
market trades through, which is what makes it able to answer the question -- and
it has already refused two passive structures on this platform.

The exit crosses deliberately. A forecast with a fifteen-minute horizon stops
being a forecast at minute sixteen, and waiting passively for a good exit is how
a directional position becomes an accidental one.

What it answered
----------------
It was run: forty-five days of fitting, twelve days held out, 800 distinct round
trips. Return -2.65%, per trade -$0.0067 +/- $0.0081, which does not clear zero.

The reason is the interesting part, and it is exact::

    model accuracy on the windows it fired on      ~65%
    direction on the trades that actually filled   50.0%

A resting entry turned a 65% signal into a coin flip. It has to: rest a bid on
an up-call and the fill arrives when the price comes *down* to it, which is when
the forecast was wrong or early, while the occasions the forecast was right are
exactly the ones that run away unfilled. Passive entry does not merely dilute
the edge, it selects against it, and the selection is close to total.

So the two doors are shut for the same reason and in opposite directions.
Crossing keeps the 65% and needs 72.6%. Resting needs only 63.5% and delivers
50%. The cheap entry and the accuracy cannot be had together, and that is a
property of resting on a directional forecast rather than of this model, this
horizon, or this venue.

The strategy is kept because the measurement is worth being able to repeat, and
because a later model that beats 72.6% crossing would not need any of this. It
is not armed.
"""

from __future__ import annotations

from kat.common.enums import MarketType
from kat.kalshi.fees import PerpFeeSchedule, perp_fee_micros
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    MIN_ORDER_QC,
    TradingStrategy,
    perp_maker_costing,
)
from kat.strategies.proposal import screen_freshness, screen_perp_market

__all__ = ["PerpModelQuoteStrategy"]

_SKIP_NO_MODEL = "no_model_probability"
_SKIP_WRONG_TARGET = "model_answers_a_different_question"
_SKIP_LOW_CONFIDENCE = "confidence_below_threshold"
_SKIP_ALREADY_QUOTING = "already_resting_in_this_market"
_SKIP_QUEUE_TOO_LONG = "queue_ahead_too_long"
_SKIP_SIZE_BELOW_MIN = "size_below_minimum"

#: The only target this strategy can read. The probability must be the chance
#: the mid is *higher* after the horizon; fed anything else -- a settlement
#: probability, a different horizon -- the number is a float and not a forecast,
#: and the trades it produces are systematically wrong rather than merely bad.
TARGET = "perp_up_15m"


class PerpModelQuoteStrategy(TradingStrategy):
    """Rests a quote on the side a forecast favours, and crosses to get out."""

    FAMILY = "model_quote"
    NAME = "Perp Model Quote"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.PERP
    DESCRIPTION = """
    Rests an entry at the touch on the side a directional model favours, so the
    entry earns the spread instead of paying it, and crosses to exit once the
    model's horizon has passed. Acts only where the model is confident, because
    the break-even accuracy is only reachable on the windows that move.
    """
    PARAM_SPACE = (
        StrategyParam(
            "min_confidence", 1_000, low=0, high=5_000, step=100,
            description="how far from even the probability must be, in bps; "
                        "1000 is a probability outside 0.40-0.60",
        ),
        StrategyParam(
            "hold_minutes", 15, low=1, high=240, step=1,
            description="the model's horizon; the exit crosses once it passes",
        ),
        StrategyParam(
            "max_queue_qc", 50_000, low=100, high=10_000_000, step=100,
            description="longest queue at our own price worth joining",
        ),
        StrategyParam(
            "min_depth_qc", 100, low=100, high=1_000_000, step=100,
            description="minimum resting size on BOTH sides across five levels",
        ),
        StrategyParam(
            "quote_notional_micros", 15_000_000, low=1_000_000,
            high=1_000_000_000, step=1_000_000,
            description="notional to rest on each entry, in micros",
        ),
    )
    REQUIRES_FEATURES = ()
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 90_000

    def warmup_ms(self) -> int:
        """The model carries its own history; this reads its output."""
        return 0

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        book, reason = screen_perp_market(
            ctx,
            max_spread_cc=1_000_000,
            min_depth_qc=self.int_param("min_depth_qc"),
        )
        if book is None or reason is not None:
            return self.skip(reason or "no_book")

        bid_cc, ask_cc = book.best_bid_cc, book.best_ask_cc
        if bid_cc is None or ask_cc is None or bid_cc >= ask_cc:
            return self.skip("no_book")
        mid_cc = (bid_cc + ask_cc) // 2

        # Exit first. A forecast with a fifteen-minute horizon has nothing to
        # say at minute sixteen, and a position held past it is no longer the
        # trade that was analysed.
        if ctx.position_qc != 0:
            age_ms = ctx.position_age_ms
            hold_ms = self.int_param("hold_minutes") * 60_000
            if age_ms is not None and age_ms >= hold_ms:
                return self.emit([self._exit(ctx, book, mid_cc)])
            return self.skip("holding_within_horizon")

        # One quote at a time: without this the strategy re-quotes every cycle
        # and accumulates a position instead of cycling one.
        if ctx.resting_qty_qc > 0:
            return self.skip(_SKIP_ALREADY_QUOTING)

        probability = ctx.model_probability
        if probability is None:
            return self.skip(_SKIP_NO_MODEL)
        # The number must be an answer to this question. Reading a settlement
        # probability, or one trained on another horizon, as a direction call is
        # how a model produces confidently wrong trades rather than no trades.
        if ctx.model_target is not None and ctx.model_target != TARGET:
            return self.skip(_SKIP_WRONG_TARGET)

        confidence_bps = int(abs(probability - 0.5) * 20_000)
        if confidence_bps < self.int_param("min_confidence"):
            return self.skip(_SKIP_LOW_CONFIDENCE)

        is_buy = probability > 0.5
        # Rest at the touch on our own side. This is the whole hypothesis: the
        # entry earns the half-spread rather than paying it, which is what
        # brings the break-even accuracy within reach of the model.
        price_cc = bid_cc if is_buy else ask_cc
        queue_qc = book.best_bid_size_qc if is_buy else book.best_ask_size_qc
        if queue_qc > self.int_param("max_queue_qc"):
            return self.skip(_SKIP_QUEUE_TOO_LONG)

        qty_qc = self._size_qc(price_cc)
        if qty_qc < MIN_ORDER_QC:
            return self.skip(_SKIP_SIZE_BELOW_MIN)

        schedule = ctx.perp_fee_schedule or PerpFeeSchedule()
        # Valued at the mid: a resting buy that fills at the bid is immediately
        # worth the mid, and that half-spread is what the forecast is added to.
        intent = self.perp_intent(
            ctx,
            is_buy=is_buy,
            qty_qc=qty_qc,
            limit_price_cc=price_cc,
            fair_value_cc=mid_cc,
            confidence=float(abs(probability - 0.5) * 2),
            costing=perp_maker_costing(
                is_long=is_buy,
                entry_cc=price_cc,
                fair_cc=mid_cc,
                qty_qc=qty_qc,
                schedule=schedule,
            ),
            reason="perp_model_quote",
            extra={
                "probability": round(probability, 4),
                "confidence_bps": confidence_bps,
                "queue_ahead_qc": queue_qc,
                "spread_cc": ask_cc - bid_cc,
                "entry_fee_micros": perp_fee_micros(
                    price_cc=price_cc, qty_qc=qty_qc,
                    schedule=schedule, is_taker=False,
                ),
            },
        )
        intent.post_only = True
        return self.emit([intent])

    # -- helpers -------------------------------------------------------------

    def _exit(self, ctx: StrategyContext, book, mid_cc: int) -> OrderIntent:
        """Cross out of the whole position; the horizon has passed."""
        is_buy = ctx.position_qc < 0
        qty_qc = abs(ctx.position_qc)
        price_cc = book.best_ask_cc if is_buy else book.best_bid_cc
        schedule = ctx.perp_fee_schedule or PerpFeeSchedule()
        intent = self.perp_intent(
            ctx,
            is_buy=is_buy,
            qty_qc=qty_qc,
            limit_price_cc=price_cc,
            fair_value_cc=mid_cc,
            confidence=0.5,
            costing=perp_maker_costing(
                is_long=is_buy,
                entry_cc=price_cc,
                fair_cc=mid_cc,
                qty_qc=qty_qc,
                schedule=schedule,
            ),
            reason="horizon_reached",
            reduce_only=True,
            extra={"position_age_ms": ctx.position_age_ms},
        )
        return intent

    def _size_qc(self, price_cc: int) -> int:
        if price_cc <= 0:
            return 0
        wanted = self.int_param("quote_notional_micros") // price_cc
        return (wanted // MIN_ORDER_QC) * MIN_ORDER_QC
