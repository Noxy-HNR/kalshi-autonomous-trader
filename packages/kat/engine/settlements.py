"""Recording how markets actually resolved.

The `settlements` table has been in the schema from the beginning and nothing
ever wrote a row to it. That is not a missing nicety: it is the reason the
platform cannot train the model it needs.

`MispricingStrategy` -- the only event strategy that consumes a model -- turns a
probability into a fair value for the contract, which is only meaningful if the
number is the chance the contract *resolves YES*. `SettlementLabel` computes
exactly that target and calls it "the target every event-contract model
ultimately wants". It needs an outcome per market. Without settlements there are
no outcomes, so no settlement model, so no usable probability, so the strategy
stands aside forever -- while the platform trains what it *can* label, whether
the mid moves 50 centicents in half an hour, which cannot be read as a fair
value without producing systematically wrong trades.

Asking the right question
-------------------------
`GET /markets?status=settled` is the obvious approach and it is useless here.
Production answers it with the same provisional shards that make `/markets`
unusable: thirty pages returned six thousand settled markets, none of which this
platform had ever priced. And the markets we *had* priced were not in it, because
they report ``finalized`` rather than ``settled``.

So this asks about our own markets instead. Every one of the forty most-quoted
tickers had resolved and carried a result; they were simply never looked up.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from kat.db.engine import Database
from kat.db.models import Settlement as SettlementRow
from kat.db.writer import AsyncWriter
from kat.kalshi.event_contracts.rest import EventContractsREST
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["SettlementRecorder"]

#: Results that mean the contract resolved one way or the other. Anything else --
#: an empty string, "void", a status we have not seen -- is not an outcome and
#: must not be labelled as one. A voided market did not resolve NO.
_OUTCOMES: dict[str, float] = {"yes": 1.0, "no": 0.0}


class SettlementRecorder:
    """Looks up how the markets we priced turned out, and writes it down.

    Only markets the platform has actually observed are asked about. The exchange
    resolves thousands of contracts we never watched; their outcomes are somebody
    else's history and have no features to attach to.
    """

    def __init__(
        self,
        rest: EventContractsREST,
        *,
        writer: AsyncWriter | None,
        database: Database | None,
        interval_s: float = 900.0,
        batch: int = 120,
    ) -> None:
        self.rest = rest
        self.writer = writer
        self.database = database
        self.interval_s = interval_s
        #: How many markets to look up per pass. Bounded so a first run against a
        #: long quote history does not spend the whole rate-limit budget at once.
        self.batch = batch
        #: Tickers resolved or known unresolved, so repeated passes do not ask
        #: the same questions. Unresolved ones are retried on the next pass.
        self._settled: set[str] = set()
        self.polls = 0
        self.written = 0
        self.unresolved = 0

    async def _pending(self) -> list[str]:
        """Observed markets with no settlement recorded yet."""
        if self.database is None:
            return []
        from kat.db.models import Quote

        async with self.database.session() as session:
            observed = {
                t for (t,) in await session.execute(select(Quote.ticker).distinct())
            }
            known = {
                t for (t,) in await session.execute(select(SettlementRow.ticker).distinct())
            }
        return sorted(observed - known - self._settled)

    async def poll_once(self) -> int:
        """One pass over a batch of unresolved markets. Returns rows written."""
        pending = await self._pending()
        if not pending:
            return 0

        self.polls += 1
        written = 0
        looked_up = 0
        for ticker in pending[: self.batch]:
            looked_up += 1
            try:
                market = await self.rest.get_market(ticker)
            except Exception as exc:
                # A market that cannot be fetched now can be fetched next pass.
                log.debug("settlements.lookup_failed", ticker=ticker, error=str(exc))
                continue

            result = str(market.result or "").strip().lower()
            if result not in _OUTCOMES:
                # Still open, or resolved to something that is not an outcome.
                self.unresolved += 1
                continue

            self._settled.add(ticker)
            if self.writer is None:
                continue
            self.writer.enqueue(
                SettlementRow(
                    ticker=ticker,
                    settled_ts=market.close_time,
                    result=result,
                    raw={
                        "status": market.status,
                        "event_ticker": market.event_ticker,
                        "last_price_cc": market.last_price_cc,
                    },
                ),
                critical=True,
            )
            written += 1

        self.written += written
        log.info(
            "settlements.recorded",
            new=written,
            total=self.written,
            looked_up=looked_up,
            still_pending=max(0, len(pending) - looked_up),
        )
        return written

    async def run(self) -> None:
        """Supervised component: poll forever, survive a bad poll.

        A settlement missed on one pass is picked up on the next, because a
        resolved market stays resolved. Nothing here is worth stopping the
        platform over.
        """
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("settlements.poll_failed", error=str(exc), exc_info=True)
            await asyncio.sleep(self.interval_s)
