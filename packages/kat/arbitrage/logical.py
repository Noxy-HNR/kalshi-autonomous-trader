"""Logically inconsistent prices on Kalshi.

Three families of inconsistency are detectable from prices alone, without any
view on the underlying question. That is what makes them worth hunting: they are
true regardless of whether the market is right about the world.

**Complementary pair.** A YES and a NO on the same market together pay exactly
$1.00 at resolution. If both can be *bought* for less than $1.00 in total, the
difference is locked in. On Kalshi this is subtle, because the exchange only
publishes bids: the ask for YES is derived from the NO bid. Buying YES at its
ask and NO at its ask means buying both sides outright, and the pair costs
``yes_ask + no_ask``.

**Mutually exclusive set.** When an event's markets are genuinely mutually
exclusive and collectively exhaustive -- exactly one resolves YES -- buying every
YES costs the sum of their asks and pays exactly $1.00. Below $1.00 that is an
arbitrage. The qualifier "genuinely" is load-bearing: the event must be flagged
mutually exclusive by the exchange, because a set that merely looks exhaustive
can have an unlisted outcome, and then buying everything pays nothing.

**Threshold ladder monotonicity.** "Above 50" cannot be less likely than
"above 60": every world where the value exceeds 60 also exceeds 50. The riskless
trade when a ladder is mispriced is to buy the **looser YES** and the **stricter
NO** -- and the direction matters enormously, because the opposite pairing is not
an arbitrage at all:

===================  ==================  ==================  ==============
Outcome              looser YES pays     stricter NO pays    total
===================  ==================  ==================  ==============
value > 60           $1.00               $0.00               **$1.00**
50 < value <= 60     $1.00               $1.00               **$2.00**
value <= 50          $0.00               $1.00               **$1.00**
===================  ==================  ==================  ==============

The payout is never below $1.00 and is $2.00 in the middle band, so buying the
pair for less than $1.00 is riskless. The mirror trade -- stricter YES plus
looser NO -- pays **nothing** in that middle band and is a directional bet that
the value lands outside it, which is exactly the trap this table exists to
document. We price the payout as $1.00 rather than its expectation, because
understating the edge is the safe direction to be wrong in.

Every candidate is handed to :mod:`kat.arbitrage.pricing`, which decides whether
it survives fees, depth and leg risk. Nothing here decides to trade.
"""

from __future__ import annotations

from dataclasses import dataclass

from kat.arbitrage.pricing import (
    PAIR_VALUE_CC,
    ArbLeg,
    PricedOpportunity,
    price_opportunity,
)
from kat.common.enums import OutcomeSide
from kat.common.ids import new_id
from kat.kalshi.fees import FeeSchedule
from kat.marketdata.book import OrderBook
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "LadderRung",
    "find_complementary_pair",
    "find_ladder_violation",
    "find_mutually_exclusive_set",
]

#: Minimum size worth attempting. Below this the fees dominate and a partial
#: fill leaves a position too small to bother unwinding.
MIN_ARB_QTY_QC = 100  # one contract


def _yes_ask(book: OrderBook, qty_qc: int) -> tuple[int, int] | None:
    """VWAP to buy ``qty_qc`` YES, and total available size."""
    depth = book.total_depth_qc(is_bid=False)
    if depth <= 0:
        return None
    vwap = book.vwap_cc(min(qty_qc, depth), is_buy=True)
    if vwap is None:
        return None
    return vwap, depth


def _no_ask(book: OrderBook, qty_qc: int) -> tuple[int, int] | None:
    """VWAP to buy ``qty_qc`` NO, and available size.

    Buying NO means lifting the YES *bid* side: a NO purchase at price ``p`` is
    matched against a YES bid at ``1 - p``. So the cost of NO is
    ``1.00 - (YES bid vwap)``, and the size available is the YES bid depth.
    """
    depth = book.total_depth_qc(is_bid=True)
    if depth <= 0:
        return None
    yes_bid_vwap = book.vwap_cc(min(qty_qc, depth), is_buy=False)
    if yes_bid_vwap is None:
        return None
    return PAIR_VALUE_CC - yes_bid_vwap, depth


def find_complementary_pair(
    ticker: str,
    book: OrderBook,
    *,
    qty_qc: int = 500,
    fee_schedule: FeeSchedule | None = None,
) -> PricedOpportunity | None:
    """Buy YES and NO on one market for less than the $1.00 they jointly pay.

    Returns the priced candidate whether or not it is worth doing, or ``None``
    when the book is too thin on one side to form the trade at all.
    """
    yes = _yes_ask(book, qty_qc)
    no = _no_ask(book, qty_qc)
    if yes is None or no is None:
        return None

    yes_price, yes_depth = yes
    no_price, no_depth = no
    size = min(qty_qc, yes_depth, no_depth)
    if size < MIN_ARB_QTY_QC:
        return None

    # Re-price at the size we can actually do, not the size we asked for.
    yes = _yes_ask(book, size)
    no = _no_ask(book, size)
    if yes is None or no is None:
        return None
    yes_price, _ = yes
    no_price, _ = no

    legs = [
        ArbLeg(
            ticker=ticker,
            side=OutcomeSide.YES,
            price_cc=yes_price,
            qty_qc=size,
            available_qc=yes_depth,
            fee_schedule=fee_schedule,
            fallback_price_cc=PAIR_VALUE_CC,
        ),
        ArbLeg(
            ticker=ticker,
            side=OutcomeSide.NO,
            price_cc=no_price,
            qty_qc=size,
            available_qc=no_depth,
            fee_schedule=fee_schedule,
            fallback_price_cc=PAIR_VALUE_CC,
        ),
    ]
    opportunity = price_opportunity(
        kind="complementary_pair",
        legs=legs,
        settlement_per_contract_micros=PAIR_VALUE_CC * 100,
        qty_qc=size,
    )
    opportunity.detail["ticker"] = ticker
    opportunity.detail["pair_cost_cc"] = yes_price + no_price
    opportunity.detail["opportunity_id"] = new_id("arb")
    return opportunity


def find_mutually_exclusive_set(
    event_ticker: str,
    books: dict[str, OrderBook],
    *,
    qty_qc: int = 500,
    fee_schedules: dict[str, FeeSchedule] | None = None,
    is_mutually_exclusive: bool,
    is_exhaustive: bool,
) -> PricedOpportunity | None:
    """Buy every YES in an exhaustive, mutually exclusive event for under $1.00.

    Refuses outright unless the event is **both** mutually exclusive and
    exhaustive. Without exclusivity two outcomes could both resolve YES and the
    payout is not $1.00; without exhaustiveness none might, and the payout is
    zero. Neither is a nuance -- either one turns a riskless trade into an
    unhedged bet on a set of outcomes we did not analyse.
    """
    if not (is_mutually_exclusive and is_exhaustive):
        log.debug(
            "arbitrage.set_skipped",
            event_ticker=event_ticker,
            mutually_exclusive=is_mutually_exclusive,
            exhaustive=is_exhaustive,
            note="a set that is not both cannot be priced as a complete set",
        )
        return None
    if len(books) < 2:
        return None

    schedules = fee_schedules or {}
    legs: list[ArbLeg] = []
    size = qty_qc
    for book in books.values():
        quote = _yes_ask(book, qty_qc)
        if quote is None:
            return None
        _, depth = quote
        size = min(size, depth)
    if size < MIN_ARB_QTY_QC:
        return None

    for ticker, book in books.items():
        quote = _yes_ask(book, size)
        if quote is None:
            return None
        price, depth = quote
        legs.append(
            ArbLeg(
                ticker=ticker,
                side=OutcomeSide.YES,
                price_cc=price,
                qty_qc=size,
                available_qc=depth,
                fee_schedule=schedules.get(ticker),
                fallback_price_cc=PAIR_VALUE_CC,
            )
        )

    opportunity = price_opportunity(
        kind="mutually_exclusive_set",
        legs=legs,
        settlement_per_contract_micros=PAIR_VALUE_CC * 100,
        qty_qc=size,
    )
    opportunity.detail["event_ticker"] = event_ticker
    opportunity.detail["set_cost_cc"] = sum(leg.price_cc for leg in legs)
    opportunity.detail["legs_count"] = len(legs)
    opportunity.detail["opportunity_id"] = new_id("arb")
    return opportunity


@dataclass(slots=True)
class LadderRung:
    """One rung of a threshold ladder, ordered by increasing strictness."""

    ticker: str
    #: The threshold itself, only used for ordering and reporting.
    threshold: float
    book: OrderBook
    fee_schedule: FeeSchedule | None = None


def find_ladder_violation(
    rungs: list[LadderRung],
    *,
    qty_qc: int = 500,
) -> PricedOpportunity | None:
    """Find a monotonicity violation in a threshold ladder.

    Rungs are supplied with their thresholds; strictness is derived from them,
    so the caller cannot get the ordering wrong. Each stricter rung logically
    implies every looser one: if the value exceeds 60 it certainly exceeds 50.

    The riskless pair is **looser YES plus stricter NO**, whose payout is $1.00
    in the outer bands and $2.00 in the middle band, so buying it for less than
    $1.00 cannot lose. See the module docstring for why the mirror trade
    (stricter YES plus looser NO) is *not* an arbitrage -- it pays nothing in the
    middle band.

    Returns the best violation found, priced. ``None`` when the ladder is
    consistent or too thin to trade.
    """
    if len(rungs) < 2:
        return None
    ordered = sorted(rungs, key=lambda r: r.threshold)

    best: PricedOpportunity | None = None
    for i in range(len(ordered) - 1):
        looser = ordered[i]
        for stricter in ordered[i + 1 :]:
            loose_yes = _yes_ask(looser.book, qty_qc)
            strict_no = _no_ask(stricter.book, qty_qc)
            if loose_yes is None or strict_no is None:
                continue
            loose_price, loose_depth = loose_yes
            strict_price, strict_depth = strict_no

            # Payout is at least $1.00 in every outcome, so any total cost below
            # $1.00 is locked in before fees.
            combined = loose_price + strict_price
            if combined >= PAIR_VALUE_CC:
                continue

            size = min(qty_qc, strict_depth, loose_depth)
            if size < MIN_ARB_QTY_QC:
                continue

            legs = [
                ArbLeg(
                    ticker=looser.ticker,
                    side=OutcomeSide.YES,
                    price_cc=loose_price,
                    qty_qc=size,
                    available_qc=loose_depth,
                    fee_schedule=looser.fee_schedule,
                    fallback_price_cc=PAIR_VALUE_CC,
                ),
                ArbLeg(
                    ticker=stricter.ticker,
                    side=OutcomeSide.NO,
                    price_cc=strict_price,
                    qty_qc=size,
                    available_qc=strict_depth,
                    fee_schedule=stricter.fee_schedule,
                    fallback_price_cc=PAIR_VALUE_CC,
                ),
            ]
            candidate = price_opportunity(
                kind="ladder_monotonicity",
                legs=legs,
                settlement_per_contract_micros=PAIR_VALUE_CC * 100,
                qty_qc=size,
            )
            candidate.detail.update(
                buy_yes_ticker=looser.ticker,
                buy_yes_threshold=looser.threshold,
                buy_no_ticker=stricter.ticker,
                buy_no_threshold=stricter.threshold,
                combined_cost_cc=combined,
                opportunity_id=new_id("arb"),
            )
            if best is None or candidate.expected_value_micros > best.expected_value_micros:
                best = candidate
    return best
