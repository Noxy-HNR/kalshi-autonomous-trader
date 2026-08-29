"""The live feature worker.

Every strategy declares ``REQUIRES_FEATURES`` and refuses to act without them.
Nothing was computing them in the running engine, so every strategy skipped on
``*_unavailable`` on every cycle: the loop turned, the risk engine was never
consulted, and the logs showed a healthy platform doing nothing. This closes
that.

Same code path as research
--------------------------
The worker maintains the rolling series a market needs and hands them to
:meth:`~kat.features.pipeline.FeaturePipeline.live_vector`, which is the same
:class:`~kat.features.compute.FeatureComputer` the training pipeline and the
backtester use. There is no live-only feature arithmetic anywhere, which is what
makes a backtest number and a live number comparable at all.

Memory
------
A long-lived process cannot keep every mid-price it has ever seen. Each series is
trimmed to a bounded window on a slow cycle, sized from the longest window any
feature in the set asks for plus a margin. Trimming to exactly the longest
window would make the oldest feature intermittently uncomputable as the boundary
moved.

Two published views
-------------------
``loop.features`` is the plain ``{name: float}`` mapping strategies read.
``loop.feature_vectors`` keeps the full :class:`FeatureVector`, timestamps and
all, because :mod:`kat.models.inference` needs the per-feature information times
to know how fresh a prediction really is. Both are replaced wholesale each cycle
rather than updated in place, so a market that stops producing features
disappears instead of going stale.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kat.common.enums import MarketType
from kat.common.timeutil import Clock
from kat.features.compute import (
    EVENT_FEATURE_SET,
    PERP_FEATURE_SET,
    Bar,
    BarSeries,
    MidSeries,
    TradePrint,
    TradeSeries,
)
from kat.features.pipeline import FeaturePipeline
from kat.features.spec import FeatureSetSpec, FeatureVector
from kat.marketdata.state import MarketState
from kat.observability.logging import get_logger
from kat.observability.metrics import metrics

log = get_logger(__name__)

__all__ = ["FeatureWorker"]

BAR_PERIOD_MS = 60_000

#: Extra history kept beyond the longest feature window. A feature asking for
#: sixty minutes needs strictly more than sixty minutes of series to be
#: computable at every instant, not exactly sixty.
TRIM_MARGIN_MS = 15 * 60_000


class _Series:
    """The rolling per-market history one feature computation needs."""

    __slots__ = ("bar_open_ms", "bar_values", "bars", "last_trade_qc", "mids", "trades")

    def __init__(self) -> None:
        self.mids = MidSeries()
        self.trades = TradeSeries(())
        self.bars: list[Bar] = []
        self.bar_open_ms: int | None = None
        self.bar_values: list[int] = []
        #: Cumulative volume at the last observation, so a delta can be taken.
        self.last_trade_qc: int | None = None

    def observe_mid(self, ts_ms: int, mid_cc: int, *, period_ms: int) -> None:
        try:
            self.mids.append(ts_ms, mid_cc)
        except Exception:
            # Out-of-order mid. The market data layer can deliver these on a
            # reconnect; dropping one point is right, and raising would take the
            # whole worker down for a recoverable hiccup.
            return

        bucket = (ts_ms // period_ms) * period_ms
        if self.bar_open_ms is None:
            self.bar_open_ms = bucket
        elif bucket != self.bar_open_ms:
            self._close_bar(period_ms)
            self.bar_open_ms = bucket
        self.bar_values.append(mid_cc)

    def _close_bar(self, period_ms: int) -> None:
        if self.bar_open_ms is None or not self.bar_values:
            self.bar_values = []
            return
        values = self.bar_values
        self.bars.append(
            Bar(
                # Close time. A bar covering 12:00:00-12:00:59 describes 12:00:59
                # and is not in our hands a moment before it.
                ts_ms=self.bar_open_ms + period_ms - 1,
                open_cc=values[0],
                high_cc=max(values),
                low_cc=min(values),
                close_cc=values[-1],
            )
        )
        self.bar_values = []

    def bar_series(self, period_ms: int) -> BarSeries:
        """Closed bars only. The bar in progress is not yet an observation."""
        return BarSeries(self.bars, period_ms=period_ms)

    def trim(self, before_ms: int) -> None:
        self.mids.trim_before(before_ms)
        self.bars = [b for b in self.bars if b.ts_ms >= before_ms]


class FeatureWorker:
    """Computes features for every market in one domain, on a fixed cadence."""

    def __init__(
        self,
        market_type: MarketType,
        state: MarketState,
        clock: Clock,
        *,
        spec: FeatureSetSpec | None = None,
        interval_s: float = 1.0,
        bar_period_ms: int = BAR_PERIOD_MS,
    ) -> None:
        self.market_type = market_type
        self.state = state
        self.clock = clock
        self.spec = spec or (
            EVENT_FEATURE_SET if market_type is MarketType.EVENT else PERP_FEATURE_SET
        )
        self.pipeline = FeaturePipeline(self.spec)
        self.interval_s = interval_s
        self.bar_period_ms = bar_period_ms

        self._series: dict[str, _Series] = {}
        self._trim_after_ms = self.spec.max_window_ms + TRIM_MARGIN_MS
        self._next_trim_ms = 0

        self.cycles = 0
        self.computed = 0
        self.unavailable = 0
        self.failures = 0
        self.last_cycle_markets = 0
        self._last_plain: dict[str, dict[str, float]] = {}

    # -- observation ---------------------------------------------------------

    def observe(self, now_ms: int) -> None:
        """Fold the current market snapshots into the rolling series."""
        for ticker in self.state.tickers():
            snapshot = self.state.get(ticker)
            if snapshot is None or snapshot.book is None:
                continue
            mid = snapshot.book.mid_cc
            if mid is None:
                continue
            series = self._series.setdefault(ticker, _Series())
            stamp = snapshot.book_updated_ms or snapshot.updated_ms or now_ms
            series.observe_mid(stamp, mid, period_ms=self.bar_period_ms)

            # Volume is cumulative on the ticker channel, so the increment is
            # the print. Without the delta every cycle would re-count the whole
            # day's volume as a single new trade.
            volume = snapshot.volume_qc
            last = snapshot.last_trade_cc
            if volume is not None and last is not None:
                previous = series.last_trade_qc
                series.last_trade_qc = volume
                if previous is not None and volume > previous:
                    series.trades = TradeSeries(
                        [
                            *series.trades.window(
                                as_of_ms=stamp, lag_ms=0, span_ms=self._trim_after_ms
                            ),
                            TradePrint(ts_ms=stamp, price_cc=last, qty_qc=volume - previous),
                        ]
                    )

    # -- computation ---------------------------------------------------------

    def compute(self, now_ms: int) -> tuple[dict[str, dict[str, float]], dict[str, FeatureVector]]:
        """Compute a vector per market. Markets that cannot be computed are absent."""
        plain: dict[str, dict[str, float]] = {}
        vectors: dict[str, FeatureVector] = {}

        for ticker, series in self._series.items():
            snapshot = self.state.get(ticker)
            if snapshot is None or snapshot.book is None:
                continue
            try:
                vector = self.pipeline.live_vector(
                    snapshot,
                    now_ms=now_ms,
                    bars=series.bar_series(self.bar_period_ms),
                    trades=series.trades,
                    mids=series.mids,
                )
            except Exception as exc:
                # A required feature that is not yet computable raises. During
                # warm-up that is the normal state of the world, so it is
                # counted rather than logged at warning level.
                self.unavailable += 1
                log.debug(
                    "features.unavailable",
                    ticker=ticker, market_type=self.market_type.value, reason=str(exc),
                )
                continue

            vectors[ticker] = vector
            plain[ticker] = {
                name: value.value
                for name, value in vector.values.items()
                if value.value is not None
            }
            self.computed += 1

        return plain, vectors

    def publish(self, loop: Any, now_ms: int) -> int:
        """One full cycle: observe, compute, publish into the loop."""
        self.cycles += 1
        self.observe(now_ms)
        plain, vectors = self.compute(now_ms)

        # Replaced wholesale, never merged. A market that has stopped producing
        # features must vanish from the loop's view rather than keep serving the
        # last value it managed to compute.
        loop.features = plain
        loop.feature_vectors = vectors
        # Kept for stats(): which feature names are warm is the question an
        # operator actually has, and it cannot be answered from counts.
        self._last_plain = plain
        self.last_cycle_markets = len(plain)

        if now_ms >= self._next_trim_ms:
            self._next_trim_ms = now_ms + 300_000
            self._trim(now_ms)

        metrics.gauge(f"features.{self.market_type.value}.markets", len(plain))
        return len(plain)

    def _trim(self, now_ms: int) -> None:
        cutoff = now_ms - self._trim_after_ms
        for ticker in list(self._series):
            if self.state.get(ticker) is None:
                # The market is gone from the watchlist; so is its history.
                del self._series[ticker]
                continue
            self._series[ticker].trim(cutoff)

    # -- component wiring ----------------------------------------------------

    async def run(self, loop: Any) -> None:
        log.info(
            "features.worker_started",
            market_type=self.market_type.value,
            feature_set=self.spec.feature_set_id,
            features=len(self.spec.specs),
            warmup_ms=self.spec.max_window_ms,
        )
        while True:
            try:
                self.publish(loop, self.clock.now_ms())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failures += 1
                log.error(
                    "features.cycle_failed",
                    market_type=self.market_type.value, error=str(exc), exc_info=True,
                )
            await asyncio.sleep(self.interval_s)

    def healthy(self) -> bool:
        """Healthy until it has run and produced nothing while markets exist.

        Reporting unhealthy during warm-up would be wrong -- there genuinely are
        no features yet -- so the check only bites once the worker has had time
        to accumulate history.
        """
        if self.cycles == 0:
            return True
        if not self.state.tickers():
            return True
        return self.last_cycle_markets > 0 or self.cycles * self.interval_s < (
            self.spec.max_window_ms / 1000
        )

    def stats(self) -> dict[str, Any]:
        """Counts, plus which features are actually warm.

        The counts alone cannot answer the question people actually have. A
        vector is published as soon as its *required* features compute, while
        several optional ones -- realised volatility needs four consecutive
        closed bars, an hour-long window needs an hour -- stay absent for a long
        while afterwards, and are filtered out of what strategies see. So
        "computed: 62, unavailable: 0" reads like everything is fine while every
        strategy that wants volatility is still declining every market.

        ``warm`` is therefore per feature: how many markets currently have a
        value for it. A name sitting at zero is the reason a strategy is
        skipping, and it is the difference between "still warming up" and
        "broken".
        """
        warm: dict[str, int] = {}
        for values in self._last_plain.values():
            for name in values:
                warm[name] = warm.get(name, 0) + 1
        expected = [s.name for s in self.spec.specs]
        return {
            "market_type": self.market_type.value,
            "feature_set_id": self.spec.feature_set_id,
            "tracked_markets": len(self._series),
            "cycles": self.cycles,
            "computed": self.computed,
            "unavailable": self.unavailable,
            "failures": self.failures,
            "last_cycle_markets": self.last_cycle_markets,
            "warm": dict(sorted(warm.items())),
            "cold": sorted(name for name in expected if name not in warm),
        }
