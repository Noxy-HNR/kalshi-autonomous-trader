"""In-memory order book.

One representation serves both domains, but they arrive differently:

* **Perpetuals** publish a conventional two-sided book: ``bids`` descending and
  ``asks`` ascending.
* **Event contracts** publish *two arrays of bids* -- ``yes_dollars`` (bids to
  buy YES) and ``no_dollars`` (bids to buy NO). Because a YES and a NO contract
  together always settle at exactly $1.00, a NO bid at price ``p`` is
  economically an offer to sell YES at ``1 - p``. The conversion is therefore

      YES ask price = $1.00 - NO bid price
      YES ask size  = NO bid size

  Getting this wrong is the single most common Kalshi integration bug: the book
  looks plausible, the spread looks plausible, and every price is inverted.
  :func:`OrderBook.from_event_contract_levels` is the only place this conversion
  happens.

Everything is stored as ``price_cc -> size_qc`` integers, so arithmetic on the
book is exact and a VWAP walk cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kat.common.fixedpoint import CC_PER_DOLLAR, notional_micros

__all__ = ["BookLevel", "OrderBook"]

#: A YES and a NO contract together settle at $1.00 == 10000 centicents.
_PAIR_VALUE_CC = CC_PER_DOLLAR


@dataclass(frozen=True, slots=True)
class BookLevel:
    price_cc: int
    size_qc: int


@dataclass(slots=True)
class OrderBook:
    """Aggregated depth for one market, in exact integer units.

    ``bids`` and ``asks`` map price to resting size. They are kept as plain
    dicts with lazily maintained sorted key lists, because the common operations
    are "best price" and "walk a few levels", not "iterate everything".
    """

    ticker: str
    bids: dict[int, int] = field(default_factory=dict)
    asks: dict[int, int] = field(default_factory=dict)
    seq: int | None = None
    ts_ms: int = 0
    _bid_prices: list[int] | None = field(default=None, repr=False)
    _ask_prices: list[int] | None = field(default=None, repr=False)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_event_contract_levels(
        cls,
        ticker: str,
        yes_levels: list[tuple[int, int]],
        no_levels: list[tuple[int, int]],
        *,
        seq: int | None = None,
        ts_ms: int = 0,
    ) -> OrderBook:
        """Build a YES-denominated book from Kalshi's two bid arrays.

        Parameters are ``(price_cc, size_qc)`` pairs already parsed from the
        fixed-point strings. ``yes_levels`` become bids directly; ``no_levels``
        are reflected through $1.00 to become asks.
        """
        book = cls(ticker=ticker, seq=seq, ts_ms=ts_ms)
        for price_cc, size_qc in yes_levels:
            if size_qc > 0:
                book.bids[price_cc] = book.bids.get(price_cc, 0) + size_qc
        for price_cc, size_qc in no_levels:
            if size_qc > 0:
                ask_cc = _PAIR_VALUE_CC - price_cc
                book.asks[ask_cc] = book.asks.get(ask_cc, 0) + size_qc
        book._invalidate()
        return book

    @classmethod
    def from_two_sided_levels(
        cls,
        ticker: str,
        bid_levels: list[tuple[int, int]],
        ask_levels: list[tuple[int, int]],
        *,
        seq: int | None = None,
        ts_ms: int = 0,
    ) -> OrderBook:
        """Build a book from a conventional two-sided feed (perpetuals)."""
        book = cls(ticker=ticker, seq=seq, ts_ms=ts_ms)
        for price_cc, size_qc in bid_levels:
            if size_qc > 0:
                book.bids[price_cc] = book.bids.get(price_cc, 0) + size_qc
        for price_cc, size_qc in ask_levels:
            if size_qc > 0:
                book.asks[price_cc] = book.asks.get(price_cc, 0) + size_qc
        book._invalidate()
        return book

    # -- mutation ------------------------------------------------------------

    def apply_delta(self, price_cc: int, delta_qc: int, *, is_bid: bool) -> None:
        """Apply one incremental level change.

        Kalshi deltas are signed changes to the resting size at a price. A level
        that reaches zero (or below, which would indicate a missed message) is
        removed.
        """
        side = self.bids if is_bid else self.asks
        new_size = side.get(price_cc, 0) + delta_qc
        if new_size > 0:
            side[price_cc] = new_size
        else:
            side.pop(price_cc, None)
        self._invalidate()

    def apply_event_contract_delta(self, price_cc: int, delta_qc: int, *, is_yes: bool) -> None:
        """Apply an event-contract delta, reflecting NO changes onto the ask side."""
        if is_yes:
            self.apply_delta(price_cc, delta_qc, is_bid=True)
        else:
            self.apply_delta(_PAIR_VALUE_CC - price_cc, delta_qc, is_bid=False)

    def _invalidate(self) -> None:
        self._bid_prices = None
        self._ask_prices = None

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self._invalidate()

    # -- queries -------------------------------------------------------------

    @property
    def bid_prices(self) -> list[int]:
        """Bid prices, best (highest) first."""
        if self._bid_prices is None:
            self._bid_prices = sorted(self.bids, reverse=True)
        return self._bid_prices

    @property
    def ask_prices(self) -> list[int]:
        """Ask prices, best (lowest) first."""
        if self._ask_prices is None:
            self._ask_prices = sorted(self.asks)
        return self._ask_prices

    @property
    def best_bid_cc(self) -> int | None:
        prices = self.bid_prices
        return prices[0] if prices else None

    @property
    def best_ask_cc(self) -> int | None:
        prices = self.ask_prices
        return prices[0] if prices else None

    @property
    def best_bid_size_qc(self) -> int:
        p = self.best_bid_cc
        return self.bids.get(p, 0) if p is not None else 0

    @property
    def best_ask_size_qc(self) -> int:
        p = self.best_ask_cc
        return self.asks.get(p, 0) if p is not None else 0

    @property
    def spread_cc(self) -> int | None:
        bid, ask = self.best_bid_cc, self.best_ask_cc
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def mid_cc(self) -> int | None:
        """Midpoint, rounded down to keep it on the integer grid.

        ``None`` when either side is empty: a one-sided book has no midpoint, and
        inventing one from the last trade is exactly how a strategy ends up
        pricing against a market that is not really there.
        """
        bid, ask = self.best_bid_cc, self.best_ask_cc
        if bid is None or ask is None:
            return None
        return (bid + ask) // 2

    @property
    def is_crossed(self) -> bool:
        """True when the best bid is at or above the best ask.

        A crossed book means we missed or misapplied a message. Callers treat it
        as a data-integrity failure and re-snapshot rather than trading on it.
        """
        bid, ask = self.best_bid_cc, self.best_ask_cc
        return bid is not None and ask is not None and bid >= ask

    def depth_qc(self, *, is_bid: bool, levels: int = 5) -> int:
        """Total resting size across the top ``levels`` on one side."""
        prices = self.bid_prices if is_bid else self.ask_prices
        side = self.bids if is_bid else self.asks
        return sum(side[p] for p in prices[:levels])

    def total_depth_qc(self, *, is_bid: bool) -> int:
        return sum((self.bids if is_bid else self.asks).values())

    def walk(self, qty_qc: int, *, is_buy: bool) -> tuple[int, int, int] | None:
        """Walk the book for ``qty_qc`` contracts.

        Returns ``(filled_qc, notional_micros, worst_price_cc)``, or ``None`` if
        the side is empty. This is the honest cost of a marketable order: it is
        what the simulator charges and what the risk engine uses to estimate
        slippage, rather than assuming a fill at the touch.
        """
        if qty_qc <= 0:
            return None
        prices = self.ask_prices if is_buy else self.bid_prices
        side = self.asks if is_buy else self.bids
        if not prices:
            return None

        remaining = qty_qc
        total_micros = 0
        worst = prices[0]
        for price in prices:
            available = side[price]
            take = min(remaining, available)
            total_micros += notional_micros(price, take)
            worst = price
            remaining -= take
            if remaining == 0:
                break
        return qty_qc - remaining, total_micros, worst

    def vwap_cc(self, qty_qc: int, *, is_buy: bool) -> int | None:
        """Average execution price in centicents for ``qty_qc``, or ``None``.

        Returns ``None`` when the book cannot fill the full size -- a partial
        answer here would understate the true cost.
        """
        result = self.walk(qty_qc, is_buy=is_buy)
        if result is None:
            return None
        filled, total, _ = result
        if filled < qty_qc or filled == 0:
            return None
        # Round against the trader: a buyer's average price rounds up.
        if is_buy:
            return -(-total // filled)
        return total // filled

    def slippage_cc(self, qty_qc: int, *, is_buy: bool) -> int | None:
        """Difference between the VWAP for ``qty_qc`` and the touch price."""
        vwap = self.vwap_cc(qty_qc, is_buy=is_buy)
        touch = self.best_ask_cc if is_buy else self.best_bid_cc
        if vwap is None or touch is None:
            return None
        return vwap - touch if is_buy else touch - vwap

    def snapshot(self, levels: int = 10) -> dict[str, object]:
        """Serialisable view for storage and the dashboard."""
        return {
            "ticker": self.ticker,
            "seq": self.seq,
            "ts_ms": self.ts_ms,
            "bids": [[p, self.bids[p]] for p in self.bid_prices[:levels]],
            "asks": [[p, self.asks[p]] for p in self.ask_prices[:levels]],
        }
