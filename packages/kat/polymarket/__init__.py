"""Polymarket US: a CFTC-regulated event-contract venue that pays its makers.

Ported after eleven structures were measured on Kalshi and all eleven lost. The
diagnosis there was never a shortage of ideas -- it was that a fee peaking at
350 basis points and a one-cent minimum tick left no room under any of them. This
venue prices the same instrument on the same quadratic formula with a maker
coefficient of -0.0125 and a tick of a tenth of a cent.

See :mod:`kat.polymarket.fees` for what that changes and
``docs/RESEARCH.md`` for what it is being asked to change it *from*.
"""

from kat.polymarket.fees import (
    PolymarketFeeSchedule,
    polymarket_fee_micros,
    polymarket_round_trip_micros,
)
from kat.polymarket.models import PolymarketMarket, parse_book
from kat.polymarket.rest import PUBLIC_GATEWAY, PolymarketREST

__all__ = [
    "PUBLIC_GATEWAY",
    "PolymarketFeeSchedule",
    "PolymarketMarket",
    "PolymarketREST",
    "parse_book",
    "polymarket_fee_micros",
    "polymarket_round_trip_micros",
]
