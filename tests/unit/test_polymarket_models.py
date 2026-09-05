"""Parsing Polymarket payloads, and the shapes that quietly return nothing.

The first survey written against this API concluded there were no two-sided
markets on the entire venue. There were 355. The book response nests everything
under ``marketData``, and reading the top level finds no ``bids`` key, returns an
empty book, and reports it as an absent market rather than as a parse failure --
a bug that looks exactly like a finding. That is why the nesting has a test.
"""

from __future__ import annotations

from decimal import Decimal

from kat.polymarket.models import PolymarketMarket, parse_book, price_cc, qty_qc

# Shaped exactly as the exchange returns it, including the marketData wrapper
# and the money objects.
BOOK = {
    "marketData": {
        "marketSlug": "tec-mlb-nlchamp-2026-09-27-atl",
        "bids": [
            {"px": {"value": "0.1540", "currency": "USD"}, "qty": "36.0000"},
            {"px": {"value": "0.1400", "currency": "USD"}, "qty": "2008.0000"},
        ],
        "offers": [
            {"px": {"value": "0.1600", "currency": "USD"}, "qty": "10.0000"},
            {"px": {"value": "0.1700", "currency": "USD"}, "qty": "500.0000"},
        ],
    }
}

MARKET = {
    "id": "7900",
    "slug": "tec-mlb-nlchamp-2026-09-27-atl",
    "question": "National League Champion",
    "category": "sports",
    "status": "MARKET_STATUS_OPEN",
    "closed": False,
    "active": True,
    "endDate": "2026-11-06T16:20:09Z",
    "orderPriceMinTickSize": 0.001,
    "minimumTradeQty": "1",
    "feeCoefficient": "0.06",
    "marketSides": [
        {"description": "Yes", "price": "0.1600"},
        {"description": "No", "price": "0.846"},
    ],
}


class TestScales:
    def test_a_tenth_of_a_cent_is_ten_centicents(self):
        """The tick that matters. Kalshi's is 100."""
        assert price_cc("0.001") == 10

    def test_a_money_object_is_unwrapped(self):
        assert price_cc({"value": "0.1540", "currency": "USD"}) == 1_540

    def test_a_bare_string_works_too(self):
        """Both shapes appear in the same payload."""
        assert price_cc("0.1540") == 1_540

    def test_absent_prices_are_none_not_zero(self):
        """Zero is a price. Missing is not, and treating it as one would put a
        contract on the book at nothing."""
        assert price_cc(None) is None
        assert price_cc("") is None
        assert price_cc({"value": None}) is None
        assert price_cc("nonsense") is None

    def test_contracts_to_centi_contracts(self):
        assert qty_qc("36.0000") == 3_600
        assert qty_qc(None) == 0


class TestBookParsing:
    def test_the_payload_nests_under_market_data(self):
        """The bug that read as a finding."""
        book = parse_book("x", BOOK)
        assert book is not None
        assert book.best_bid_cc == 1_540
        assert book.best_ask_cc == 1_600

    def test_sizes_survive(self):
        book = parse_book("x", BOOK)
        assert book is not None
        assert book.best_bid_size_qc == 3_600
        assert book.best_ask_size_qc == 1_000

    def test_levels_are_ordered_best_first(self):
        """The exchange does not promise an order, and a book sorted the wrong
        way makes the worst price look like the touch."""
        scrambled = {"marketData": {
            "bids": [{"px": "0.10", "qty": "5"}, {"px": "0.15", "qty": "5"}],
            "offers": [{"px": "0.20", "qty": "5"}, {"px": "0.16", "qty": "5"}],
        }}
        book = parse_book("x", scrambled)
        assert book is not None
        assert book.best_bid_cc == 1_500
        assert book.best_ask_cc == 1_600

    def test_a_one_sided_book_is_none_not_empty(self):
        """It cannot be priced, and inventing the missing half is how a strategy
        quotes against a market that is not there."""
        assert parse_book("x", {"marketData": {"bids": [{"px": "0.1", "qty": "1"}],
                                               "offers": []}}) is None
        assert parse_book("x", {"marketData": {"bids": [], "offers": []}}) is None
        assert parse_book("x", {}) is None

    def test_zero_size_levels_are_dropped(self):
        assert parse_book("x", {"marketData": {
            "bids": [{"px": "0.10", "qty": "0"}],
            "offers": [{"px": "0.20", "qty": "5"}],
        }}) is None


class TestMarketParsing:
    def test_the_fee_schedule_comes_from_the_market(self):
        m = PolymarketMarket.parse(MARKET)
        assert m.fees.exact is True
        assert m.fees.taker_theta == Decimal("0.06")
        assert m.fees.pays_makers is True

    def test_the_fine_tick_is_read(self):
        assert PolymarketMarket.parse(MARKET).tick_size_cc == 10

    def test_a_market_without_a_tick_is_assumed_coarse(self):
        """Assuming the finest tick on the venue would let a strategy quote at
        an increment the exchange rejects."""
        payload = {k: v for k, v in MARKET.items() if k != "orderPriceMinTickSize"}
        assert PolymarketMarket.parse(payload).tick_size_cc == 100

    def test_tradeability_needs_active_and_not_closed(self):
        assert PolymarketMarket.parse(MARKET).is_tradeable is True
        assert PolymarketMarket.parse({**MARKET, "closed": True}).is_tradeable is False
        assert PolymarketMarket.parse({**MARKET, "active": False}).is_tradeable is False

    def test_outcomes_carry_their_prices(self):
        m = PolymarketMarket.parse(MARKET)
        assert m.outcomes == [("Yes", 1_600), ("No", 8_460)]

    def test_a_sparse_payload_does_not_raise(self):
        """The listing endpoint omits fields the detail endpoint returns."""
        m = PolymarketMarket.parse({"slug": "s", "id": "1"})
        assert m.slug == "s"
        assert m.is_tradeable is False
        assert m.fees.exact is False


class TestMarketLookupUsesTheRightRoute:
    """``/v1/markets/{slug}`` is not a route and 404s for every market.

    The book and bbo *are* addressed that way -- ``/v1/markets/{slug}/book``
    works -- which is what made the wrong guess look reasonable and why every
    call in an analysis script failed identically until one was isolated.
    """

    def test_it_queries_the_listing_endpoint_with_a_slug_filter(self):
        import asyncio

        from kat.polymarket.rest import PolymarketREST

        seen: dict[str, object] = {}

        class _Fake(PolymarketREST):
            async def _get(self, path, params=None):
                seen["path"], seen["params"] = path, params
                return {"markets": [dict(MARKET)]}

        got = asyncio.run(_Fake().get_market("some-slug"))
        assert seen["path"] == "/v1/markets"
        assert (seen["params"] or {}).get("slug") == "some-slug"
        assert got is not None and got.slug == MARKET["slug"]

    def test_an_unknown_slug_is_none_rather_than_a_raise(self):
        import asyncio

        from kat.polymarket.rest import PolymarketREST

        class _Empty(PolymarketREST):
            async def _get(self, path, params=None):
                return {"markets": []}

        assert asyncio.run(_Empty().get_market("nope")) is None
