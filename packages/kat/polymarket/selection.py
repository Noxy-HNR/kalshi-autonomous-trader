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
