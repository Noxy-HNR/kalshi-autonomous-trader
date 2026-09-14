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
import bisect
from datetime import datetime, timedelta

from sqlalchemy import select

from kat.common.timeutil import utcnow_aware
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

#: How far back each pass re-reads quotes for markets it has not yet seen.
#:
#: Asking for every distinct ticker in the quotes table, every fifteen minutes,
#: costs more each time the table grows, and the table only grows. Reading since
#: the previous pass is enough; the overlap covers rows written late, stamped with
#: the time they were observed rather than the time they landed.
_OBSERVED_OVERLAP = timedelta(hours=1)

#: Passes between full rescans of the quotes table.
#:
#: The incremental read filters on when a quote was observed, because that column
#: is indexed and the table is the largest in the schema. Measured over 48 hours
#: the worst gap between a quote's observation and its row landing was 2 minutes
#: 40 seconds, well inside the overlap -- but a stalled database writer holds rows
#: for as long as it is stalled, and a quote that lands after its window has
#: passed would never be read. A full rescan once a day at the default fifteen
#: minute interval bounds that miss to a day instead of forever.
_FULL_RESCAN_EVERY = 96


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
        #: Every market ever quoted, built up pass by pass rather than re-read.
        self._observed: set[str] = set()
        self._observed_since: datetime | None = None
        self._scans = 0
        #: Where the last pass stopped, so the next one carries on from there.
        self._after: str | None = None
        self.polls = 0
        self.written = 0
        self.unresolved = 0

    async def _pending(self) -> list[str]:
        """Observed markets with no settlement recorded yet."""
        if self.database is None:
            return []
        from kat.db.models import Quote

        started = utcnow_aware()
        query = select(Quote.ticker).distinct()
        full = self._observed_since is None or self._scans % _FULL_RESCAN_EVERY == 0
        self._scans += 1
        if not full:
            query = query.where(Quote.ts > self._observed_since)
        async with self.database.session() as session:
            self._observed.update(t for (t,) in await session.execute(query))
            known = {
                t for (t,) in await session.execute(select(SettlementRow.ticker).distinct())
            }
        self._observed_since = started - _OBSERVED_OVERLAP
        return sorted(self._observed - known - self._settled)

    async def poll_once(self) -> int:
        """One pass over a batch of unresolved markets. Returns rows written."""
        pending = await self._pending()
        if not pending:
            return 0

        self.polls += 1
        written = 0
        looked_up = 0
        # Carry on from where the last pass stopped, wrapping at the end.
        #
        # Every pass used to take the same alphabetical head of the backlog. A
        # market still open is retried, not settled, so once the first 120 were
        # long-dated every pass asked about those 120 and nothing after them was
        # ever reached -- reproduced as 11,520 lookups of 120 markets in a day
        # and 500 resolved markets never asked about, and seen live as a
        # still_pending count that only rose.
        start = bisect.bisect_right(pending, self._after) if self._after is not None else 0
        window = pending[start : start + self.batch]
        if len(window) < self.batch and start > 0:
            window += pending[: min(start, self.batch - len(window))]
        for ticker in window:
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

        if window:
            self._after = window[-1]
        self.written += written
        log.info(
            "settlements.recorded",
            new=written,
            total=self.written,
            looked_up=looked_up,
            # Markets still without a recorded outcome after this pass -- not the
            # ones this pass happened not to reach, which is what this used to say.
            still_pending=max(0, len(pending) - written),
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
