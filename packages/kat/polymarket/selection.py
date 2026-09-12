"""Choosing which markets are worth watching closely.

The collector can follow about sixty markets at a resolution fine enough to
measure a fill, out of four thousand open ones. Which sixty decides what every
later question can be asked of, so the choice is here, tested, rather than in the
script that happens to call it.

The ranking is the one thing the Kalshi work established and then failed to act
on. That survey measured spread and queue depth separately, found 879 markets
trading over a thousand contracts a day, and still produced nine round trips in
seventy-two hours -- because it never divided one by the other. Fills go in time
order: a resting order is reached only once everything already at its price has
traded. A hundred contracts an hour behind a queue of ten thousand never reaches
us; ten an hour behind a queue of five reaches us twice.

So markets are ranked on edge per contract of queue ahead, not on edge.
"""

from __future__ import annotations

from typing import Any

from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT
from kat.polymarket.fees import polymarket_fee_micros

__all__ = ["maker_edge_cc", "rank_by_reachability"]


def maker_edge_cc(market: Any, book: Any) -> int | None:
    """What one passive round trip earns here: the spread, plus both rebates.

    Returns centicents per contract, or None when the book cannot be priced.

    Fees are signed on this venue -- a maker rebate arrives negative -- so
    subtracting it *adds* the income. That is the whole reason the platform is
    here, and it is also the line that would silently invert if anyone wrapped
    the fee in ``abs()``.
    """
    bid, ask = book.best_bid_cc, book.best_ask_cc
    if bid is None or ask is None or not 0 < bid < ask < CC_PER_DOLLAR:
        return None
    mid = (bid + ask) // 2
    fee_cc = polymarket_fee_micros(
        price_cc=mid,
        qty_qc=QC_PER_CONTRACT,
        schedule=getattr(market, "fees", None),
        is_taker=False,
    ) // 100
    return (ask - bid) - 2 * fee_cc


def rank_by_reachability(
    candidates: list[tuple[Any, int, int]], limit: int
) -> list[str]:
    """Pick the ``limit`` markets whose edge is most likely to be collected.

    ``candidates`` are ``(market, edge_cc, queue_qc)``. Markets whose round trip
    does not pay are dropped outright: following one to see whether it later
    starts paying spends a slot that a paying market could have used.

    A queue shorter than one contract scores as one contract. An empty touch
    means we would be first in line, which is the best case and not an infinite
    one, and dividing by zero to express that would rank a market on a rounding
    artifact.

    This ranks on reachability alone and is therefore only half a selector --
    see :func:`rank_tradeable`, which is what the collector should use. Kept
    because it is the honest expression of "which queue could we reach", and
    because separating that from "which markets trade" is what made the
    difference visible.
    """
    scored = [
        (edge / max(queue, QC_PER_CONTRACT), market.slug)
        for market, edge, queue in candidates
        if edge > 0
    ]
    # Slug breaks ties so the follow set is stable between sweeps: an unstable
    # set silently restarts each market's history every six hours.
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [slug for _, slug in scored[:limit]]


#: The narrowest edge worth spending fine resolution on, in centicents.
#:
#: Measured over 277 fills a continuous touch quote would have taken here: the
#: five-minute mark-out from our own fill price runs -300 +- 104cc where the
#: spread is under two cents and -151 +- 111 under four, turning positive at
#: +167 +- 76 only above it. A market under four cents is one the strategy
#: refuses to quote, so following it closely spends two requests a pass on data
#: nothing will ever act on.
#:
#: This is ``PassiveQuoteStrategy``'s ``min_spread_cc`` -- the same number for
#: the same reason -- and the two have to move together. They are not wired to
#: each other on purpose: a data collector that imported a strategy's parameters
#: would collect only what today's strategy wants, and the sweep that feeds every
#: other measurement here is the thing that would quietly narrow.
MIN_FOLLOW_EDGE_CC = 400


def rank_tradeable(
    candidates: list[tuple[Any, int, int, int | None]], limit: int,
    *, min_edge_cc: int = 0,
) -> list[str]:
    """Rank on reachability, but only among markets that actually trade.

    ``candidates`` are ``(market, edge_cc, queue_qc, traded_qc)``, where
    ``traded_qc`` is the venue's cumulative traded counter -- None when the
    market has never reported one.

    Ranking on reachability alone was measurably wrong, and wrong in a way that
    hid its own error. It favours short queues; a short queue usually means few
    people are quoting; and few quoters correlates with few takers. The first
    follow set built that way reported **zero trades in 297 intervals**, while a
    broad unranked sample of the same venue found 15% of counter-reporting
    markets trading and 1,473 contracts an hour. The ranking had selected the
    markets most likely to report nothing.

    Requiring a non-zero counter fixes it without discarding the insight, which
    was sound: measured against the markets that did trade, a short queue is
    exactly what produces fills. Six contracts of queue against 239 traded an
    hour is twenty fills an hour; 102,299 of queue against 63 is none. Flow is
    worthless behind a deep queue and a short queue is worthless without flow,
    and only markets with both are worth a slot.
    """
    tradeable = [
        (edge / max(queue, QC_PER_CONTRACT), edge, market.slug)
        for market, edge, queue, traded in candidates
        if edge > 0 and traded
    ]
    # Quotable first, then the rest, each by reachability. Two tiers rather than
    # a filter: a floor on its own would shrink the follow set on a quiet board
    # and leave slots unspent, and an unfilled slot collects nothing at all.
    preferred = sorted(
        (pair for pair in tradeable if pair[1] >= min_edge_cc),
        key=lambda pair: (-pair[0], pair[2]),
    )
    rest = sorted(
        (pair for pair in tradeable if pair[1] < min_edge_cc),
        key=lambda pair: (-pair[0], pair[2]),
    )
    return [slug for _, _, slug in (preferred + rest)[:limit]]
