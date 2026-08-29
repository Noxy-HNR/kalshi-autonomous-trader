"""The snapshot parser must recognise the field names the exchange actually uses.

Kalshi has shipped three names for the same array. Matching none of them is
silent: the lookup returns None, `_levels(None)` is an empty list, and
`from_event_contract_levels` accepts that as a market with no orders on either
side. The snapshot applies, nothing raises, and the book then rebuilds from
whatever deltas arrive next -- a partial, usually one-sided view of a market that
is quoted on both sides.

Live against production that showed as 4 of 60 markets having a midpoint, book
ages of twenty-five minutes, and a dashboard that looked like it was capturing
nothing at all.
"""

from __future__ import annotations

import pytest

from kat.common.enums import MarketType
from kat.common.timeutil import SystemClock
from kat.engine.ingestion import MarketDataIngestor, _first
from kat.marketdata.state import MarketState


class _NullWS:
    connected = True

    def on(self, *_a: object, **_k: object) -> None:
        pass


def _ingestor() -> MarketDataIngestor:
    return MarketDataIngestor(
        market_type=MarketType.EVENT,
        ws=_NullWS(),  # type: ignore[arg-type]
        state=MarketState(MarketType.EVENT),
        clock=SystemClock(),
        writer=None,
        persist_quotes=False,
    )


@pytest.mark.parametrize(
    ("yes_key", "no_key"),
    [
        ("yes_dollars_fp", "no_dollars_fp"),  # production, as captured live
        ("yes_dollars", "no_dollars"),
        ("yes", "no"),
    ],
)
def test_every_schema_variant_produces_a_two_sided_book(yes_key: str, no_key: str) -> None:
    """A YES bid at 10 and a NO bid at 9980 is a 10/20 market."""
    book = _ingestor()._book_from_snapshot(
        "T", {yes_key: [["0.0010", "77208.59"]], no_key: [["0.9980", "5109780.00"]]}
    )
    assert book is not None
    assert book.best_bid_cc == 10
    assert book.best_ask_cc == 20
    assert book.mid_cc == 15
    assert book.spread_cc == 10


def test_an_unrecognised_schema_does_not_silently_produce_an_empty_book() -> None:
    """The failure this whole module is about.

    If Kalshi renames these again, the book must come back empty *visibly* --
    with no bid and no ask -- rather than half-built from later deltas. This
    test exists to fail loudly the next time the name changes.
    """
    book = _ingestor()._book_from_snapshot(
        "T", {"yes_renamed_again": [["0.0010", "1"]], "no_renamed_again": [["0.9980", "1"]]}
    )
    assert book is not None
    assert book.best_bid_cc is None
    assert book.best_ask_cc is None


def test_first_prefers_an_empty_list_over_a_later_name() -> None:
    """An empty list is a legitimate value: a market quoted on one side only.

    Chaining with ``or`` would skip past it to the next schema's name and read
    the book with the wrong one.
    """
    assert _first({"a": [], "b": [["0.5", "1"]]}, "a", "b") == []
    assert _first({"b": [["0.5", "1"]]}, "a", "b") == [["0.5", "1"]]
    assert _first({}, "a", "b") is None


def test_one_sided_markets_stay_one_sided() -> None:
    """Only a YES bid means a bid and no ask, and that must survive parsing."""
    book = _ingestor()._book_from_snapshot("T", {"yes_dollars_fp": [["0.3000", "500"]]})
    assert book is not None
    assert book.best_bid_cc == 3000
    assert book.best_ask_cc is None
