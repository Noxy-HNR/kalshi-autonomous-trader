"""An absent quote must not be mistaken for a price at the boundary.

Kalshi reports a market with no orders as a full-width book -- bid ``$0.0000``,
ask ``$1.0000`` -- rather than as a null. Those parse into perfectly valid
integers, so the ``is not None`` test the watchlist used accepted every market
on the exchange: 1,000 of 1,000 "tradeable", and 57 of 60 subscription slots
spent on markets that had no orders in them at all.
"""

from __future__ import annotations

import pytest

from kat.kalshi.event_contracts.models import EventMarket


def market(bid: str | None, ask: str | None, status: str = "active") -> EventMarket:
    payload: dict = {"ticker": "KX-TEST", "status": status}
    if bid is not None:
        payload["yes_bid_dollars"] = bid
    if ask is not None:
        payload["yes_ask_dollars"] = ask
    return EventMarket.parse(payload)


class TestHasLiveQuote:
    def test_the_full_width_book_kalshi_sends_for_no_orders(self):
        """The exact payload the demo exchange returns for an empty market."""
        m = market("0.0000", "1.0000")
        # It parses to real integers -- that is the trap.
        assert m.yes_bid_cc == 0
        assert m.yes_ask_cc == 10_000
        assert m.yes_bid_cc is not None and m.yes_ask_cc is not None
        # And it is still not a quote.
        assert m.has_live_quote is False

    def test_a_genuine_two_sided_quote(self):
        assert market("0.4200", "0.4500").has_live_quote is True

    @pytest.mark.parametrize(
        ("bid", "ask", "why"),
        [
            ("0.0000", "0.4500", "no bid"),
            ("0.4200", "1.0000", "no ask"),
            ("0.0000", "1.0000", "neither side"),
            ("0.4500", "0.4200", "crossed"),
            ("0.4200", "0.4200", "locked"),
        ],
    )
    def test_degenerate_books_are_not_quotes(self, bid, ask, why):
        assert market(bid, ask).has_live_quote is False, why

    def test_a_missing_side_is_not_a_quote(self):
        assert market(None, "0.4500").has_live_quote is False
        assert market("0.4200", None).has_live_quote is False
        assert market(None, None).has_live_quote is False

    def test_a_quote_one_tick_inside_each_boundary_counts(self):
        """The bound is exclusive, not a wide dead zone."""
        assert market("0.0001", "0.9999").has_live_quote is True

    def test_quoting_is_independent_of_being_active(self):
        """Two different questions; the watchlist asks both."""
        closed = market("0.4200", "0.4500", status="closed")
        assert closed.has_live_quote is True
        assert closed.is_tradeable is False
