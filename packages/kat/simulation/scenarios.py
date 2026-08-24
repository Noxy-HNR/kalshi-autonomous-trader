"""Generated market histories, and what they are and are not good for.

Everything this module produces is stamped :data:`DataQuality.SYNTHETIC`, which
:class:`~kat.simulation.engine.EventFeed` refuses to mix with recorded data. That
is the whole safety property: a number measured here can never be quietly
presented as a number measured on the real exchange.

What generated history is for
-----------------------------
Proving that the machinery works. A strategy that cannot make money on a market
that is *by construction* mean-reverting has a bug, not a weak edge. A backtest
that reports a Sharpe on a pure random walk has a bug too. These scenarios exist
so the simulator, the fill model, the ledger and the gates can be exercised and
falsified before any real capital or any real data is involved.

What it is emphatically not for
-------------------------------
Deciding that a strategy works. Generated data has the statistical structure it
was told to have, and a strategy tuned against it has been tuned against an
assumption. :mod:`kat.discovery.lifecycle` treats a synthetic run as evidence
that the pipeline ran, never as evidence that an edge exists; promotion past
BACKTEST requires recorded history.

Realism that is deliberately included
-------------------------------------
The generators reproduce the specific features of Kalshi's microstructure that
change a strategy's answer: prices live on a tick grid and are clamped to
``[1, 9999]`` centicents because a contract cannot trade at zero or one dollar;
books are two-sided with a spread that widens with volatility; quantities are
finite so a large order walks the book; and every observation carries a
transport lag, so nothing is knowable at the instant it happens.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

from kat.common.enums import DataQuality
from kat.simulation.engine import (
    SimEvent,
    book_event,
    settlement_event,
    ticker_event,
    trade_event,
)

__all__ = [
    "ScenarioSpec",
    "drifting_market",
    "generate_market_series",
    "generate_scenario",
    "market_definitions",
    "mean_reverting_market",
    "overreaction_market",
    "random_walk_market",
    "settling_market",
    "trending_market",
]

#: Kalshi contracts trade in [1, 99] cents. A contract at zero or a dollar has
#: resolved and is no longer a market.
MIN_PRICE_CC = 100
MAX_PRICE_CC = 9_900

#: Transport delay applied to every generated observation. Nothing is knowable
#: at the instant it happens, and a generator that pretended otherwise would
#: make the leakage detector untestable.
DEFAULT_LAG_MS = 120


@dataclass(slots=True)
class ScenarioSpec:
    """The parameters of one generated history.

    Recorded on the result so a run is reproducible from its spec alone. The
    seed is part of the spec for the same reason: "the mean-reverting scenario"
    is not a reproducible claim, but "the mean-reverting scenario at seed 7" is.
    """

    kind: str = "mean_reverting"
    ticker: str = "SIM-MARKET"
    event_ticker: str = "SIM-EVENT"
    #: Simulated wall-clock duration.
    duration_ms: int = 6 * 3_600_000
    #: Gap between successive book snapshots.
    step_ms: int = 5_000
    start_price_cc: int = 5_000
    #: Standard deviation of one step, in centicents.
    volatility_cc: float = 40.0
    #: Half-spread at zero volatility, in centicents.
    #:
    #: Calibrated against the live exchange, not guessed. Sampling 1,395 open
    #: markets carrying two-sided quotes gave these median spreads by price
    #: band, in centicents::
    #:
    #:     mid 0.02-0.15   p25 200   median 445   p75  680
    #:     mid 0.15-0.35   p25 300   median 600   p75  800
    #:     mid 0.35-0.65   p25 500   median 900   p75 2600
    #:     mid 0.65-0.85   p25 600   median 900   p75 1100
    #:     mid 0.85-0.98   p25 590   median 900   p75 1000
    #:
    #: The default is the p25 of the cheap band -- a *good* market, not a
    #: typical one -- because the interesting question is whether a strategy
    #: works where conditions favour it. Anyone modelling a typical market
    #: should raise this, and will find most strategies stop trading, which is
    #: the correct answer and the reason these numbers are recorded here rather
    #: than chosen to flatter a backtest.
    base_half_spread_cc: int = 100
    #: Contracts resting at the touch, in centi-contracts.
    top_of_book_qc: int = 5_000
    depth_levels: int = 5
    #: Pull toward ``start_price_cc`` per step, as a fraction of the gap. Only
    #: used by the mean-reverting generator.
    reversion: float = 0.08
    #: Per-step chance of a sharp jump, used by the overreaction generator.
    shock_probability: float = 0.004
    #: Size of that jump in centicents, before the reversion pulls it back.
    shock_cc: float = 350.0
    #: Constant per-step drift in centicents. Only used by the trending
    #: generator, where it is what makes the trend a trend.
    drift_cc: float = 3.0
    lag_ms: int = DEFAULT_LAG_MS
    #: When the market stops trading, measured from the start of the scenario.
    #: Event contracts always have one, and the term-structure features are
    #: undefined without it -- a generated history that omitted it would produce
    #: a feed no strategy could act on. ``None`` for perpetuals, which do not
    #: expire.
    close_offset_ms: int | None = None
    seed: int = 0
    #: Emit a settlement at the end, resolving on where the price finished.
    settle: bool = False
    #: Perpetuals only: funding rate as a raw per-interval fraction. See
    #: ``MarketSnapshot.funding_rate`` -- this is not basis points.
    funding_rate: float | None = None
    funding_interval_hours: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ticker": self.ticker,
            "duration_ms": self.duration_ms,
            "step_ms": self.step_ms,
            "start_price_cc": self.start_price_cc,
            "volatility_cc": self.volatility_cc,
            "reversion": self.reversion,
            "drift_cc": self.drift_cc,
            "lag_ms": self.lag_ms,
            "close_offset_ms": self.close_offset_ms,
            "seed": self.seed,
            "settle": self.settle,
            "funding_rate": self.funding_rate,
        }


@dataclass(slots=True)
class _Walker:
    """Shared price-path machinery, so the generators differ only in their step."""

    spec: ScenarioSpec
    rng: random.Random
    price_cc: float = 0.0
    events: list[SimEvent] = field(default_factory=list)
    seq: int = 0

    def __post_init__(self) -> None:
        self.price_cc = float(self.spec.start_price_cc)

    def _clamp(self, value: float) -> float:
        return max(float(MIN_PRICE_CC), min(float(MAX_PRICE_CC), value))

    def _half_spread_cc(self, recent_move: float) -> int:
        """Spread widens with realised movement, as a real book's does.

        A fixed spread makes every strategy look better than it is: the cost of
        crossing is the single largest term in a Kalshi strategy's P&L, and
        holding it constant while volatility changes removes the main reason a
        strategy should stand aside.
        """
        widen = min(3.0, 1.0 + abs(recent_move) / max(1.0, self.spec.volatility_cc))
        return max(10, int(self.spec.base_half_spread_cc * widen))

    def emit(self, now_ms: int, recent_move: float) -> None:
        mid = self._clamp(self.price_cc)
        half = self._half_spread_cc(recent_move)

        bids: list[tuple[int, int]] = []
        asks: list[tuple[int, int]] = []
        for level in range(self.spec.depth_levels):
            # Depth thins as it goes out, so a large order pays progressively
            # worse prices instead of filling the whole clip at the touch.
            size = max(100, int(self.spec.top_of_book_qc * (0.6**level)))
            bid_cc = int(self._clamp(mid - half - level * half))
            ask_cc = int(self._clamp(mid + half + level * half))
            if bid_cc < ask_cc:
                bids.append((bid_cc, size))
                asks.append((ask_cc, size))

        if not bids or not asks:  # pragma: no cover - only at the clamp bounds
            return

        self.seq += 1
        self.events.append(
            book_event(
                self.spec.ticker,
                info_ts_ms=now_ms,
                lag_ms=self.spec.lag_ms,
                bids=bids,
                asks=asks,
                quality=DataQuality.SYNTHETIC,
                seq=self.seq,
                event_ticker=self.spec.event_ticker,
            )
        )

        # A print roughly every fourth snapshot, on one side or the other.
        if self.rng.random() < 0.25:
            self.seq += 1
            side_price = bids[0][0] if self.rng.random() < 0.5 else asks[0][0]
            self.events.append(
                trade_event(
                    self.spec.ticker,
                    info_ts_ms=now_ms,
                    price_cc=side_price,
                    qty_qc=self.rng.randint(100, 2_000),
                    lag_ms=self.spec.lag_ms,
                    quality=DataQuality.SYNTHETIC,
                    seq=self.seq,
                )
            )

        if self.spec.funding_rate is not None:
            self.seq += 1
            self.events.append(
                ticker_event(
                    self.spec.ticker,
                    info_ts_ms=now_ms,
                    lag_ms=self.spec.lag_ms,
                    mark_price_cc=int(mid),
                    funding_rate=self.spec.funding_rate,
                    next_funding_ms=now_ms + int(self.spec.funding_interval_hours * 3_600_000),
                    quality=DataQuality.SYNTHETIC,
                    seq=self.seq,
                )
            )

    def finish(self, now_ms: int) -> list[SimEvent]:
        if self.spec.settle:
            self.seq += 1
            self.events.append(
                settlement_event(
                    self.spec.ticker,
                    info_ts_ms=now_ms,
                    outcome_yes=self.price_cc >= 5_000,
                    lag_ms=self.spec.lag_ms,
                    quality=DataQuality.SYNTHETIC,
                    seq=self.seq,
                )
            )
        return self.events


def _walk(spec: ScenarioSpec, step: Any) -> list[SimEvent]:
    """Run ``step(walker, previous_move)`` over the scenario's duration."""
    walker = _Walker(spec=spec, rng=random.Random(spec.seed))
    now_ms = 0
    move = 0.0
    while now_ms <= spec.duration_ms:
        walker.emit(now_ms, move)
        before = walker.price_cc
        step(walker)
        walker.price_cc = walker._clamp(walker.price_cc)
        move = walker.price_cc - before
        now_ms += spec.step_ms
    return walker.finish(now_ms)


def mean_reverting_market(spec: ScenarioSpec) -> list[SimEvent]:
    """An Ornstein-Uhlenbeck price pulled toward its starting level.

    The scenario a mean-reversion strategy is *supposed* to profit from. If it
    does not, the strategy or the plumbing is broken -- which is the only
    conclusion this scenario licenses.
    """

    def step(w: _Walker) -> None:
        pull = (spec.start_price_cc - w.price_cc) * spec.reversion
        w.price_cc += pull + w.rng.gauss(0.0, spec.volatility_cc)

    return _walk(spec, step)


def trending_market(spec: ScenarioSpec) -> list[SimEvent]:
    """A steady drift with noise on top.

    Kept distinct from :func:`drifting_market` because the interesting property
    is that lag-1 autocorrelation of the returns is near zero here: a constant
    step is not serially correlated, which is why trend detection has to measure
    net movement against path length rather than autocorrelation.
    """

    def step(w: _Walker) -> None:
        w.price_cc += spec.drift_cc + w.rng.gauss(0.0, spec.volatility_cc)

    return _walk(spec, step)


def drifting_market(spec: ScenarioSpec) -> list[SimEvent]:
    """Drift that reverses direction partway through.

    A strategy that only works on the first half is fitted to the first half,
    and a walk-forward split over this scenario will say so.
    """
    midpoint = spec.duration_ms // 2

    def step(w: _Walker) -> None:
        elapsed = w.seq * spec.step_ms
        direction = 1.0 if elapsed < midpoint else -1.0
        w.price_cc += direction * spec.drift_cc + w.rng.gauss(0.0, spec.volatility_cc)

    return _walk(spec, step)


def random_walk_market(spec: ScenarioSpec) -> list[SimEvent]:
    """A driftless random walk: the null hypothesis.

    Nothing is predictable here by construction. A strategy that reports an edge
    on this scenario has found one in the noise, and the honest response is to
    treat the strategy as broken rather than the scenario as lucky. This is the
    single most useful generator in the module.
    """

    def step(w: _Walker) -> None:
        w.price_cc += w.rng.gauss(0.0, spec.volatility_cc)

    return _walk(spec, step)


def settling_market(spec: ScenarioSpec) -> list[SimEvent]:
    """A price that converges to its resolution as expiry approaches.

    Reproduces the one structural feature of an event contract that has no
    analogue in a continuous market: uncertainty collapses on a known schedule,
    so the same spread costs more as expiry nears and a position held to the end
    pays zero or one dollar rather than a mark.
    """
    outcome = 1.0 if random.Random(spec.seed).random() < 0.5 else 0.0
    target = MAX_PRICE_CC if outcome else MIN_PRICE_CC
    total_steps = max(1, spec.duration_ms // spec.step_ms)

    def step(w: _Walker) -> None:
        progress = min(1.0, w.seq / total_steps)
        # Convergence accelerates: most of the resolution happens near the end,
        # which is where the interesting trading is.
        pull = (target - w.price_cc) * (progress**2) * 0.15
        noise = w.rng.gauss(0.0, spec.volatility_cc * (1.0 - progress * 0.7))
        w.price_cc += pull + noise

    spec.settle = True
    events = _walk(spec, step)
    # Replace the price-derived settlement with the outcome the path converged
    # to, so the resolution and the path agree.
    return [
        settlement_event(
            spec.ticker,
            info_ts_ms=e.info_ts_ms,
            outcome_yes=bool(outcome),
            lag_ms=spec.lag_ms,
            quality=DataQuality.SYNTHETIC,
            seq=e.seq,
        )
        if e.kind.value == "settlement"
        else e
        for e in events
    ]


def overreaction_market(spec: ScenarioSpec) -> list[SimEvent]:
    """Mean reversion punctuated by sharp, unconfirmed spikes that retrace.

    The pattern the fade strategy claims to exploit, constructed on purpose: a
    price jumps several hundred centicents in one step on no volume, stalls, and
    is pulled back by the same reversion that governs the rest of the path.

    That makes this a falsification test rather than a demonstration. If the
    strategy cannot profit on a market that was *built* to contain its edge, the
    strategy or the plumbing is broken -- and that is the only thing a positive
    result here licenses anyone to conclude. It says nothing whatever about
    whether the pattern exists on the real exchange.
    """

    def step(w: _Walker) -> None:
        pull = (spec.start_price_cc - w.price_cc) * spec.reversion
        shock = 0.0
        if w.rng.random() < spec.shock_probability:
            shock = spec.shock_cc * (1.0 if w.rng.random() < 0.5 else -1.0)
        w.price_cc += pull + shock + w.rng.gauss(0.0, spec.volatility_cc)

    return _walk(spec, step)

_GENERATORS = {
    "mean_reverting": mean_reverting_market,
    "overreaction": overreaction_market,
    "trending": trending_market,
    "drifting": drifting_market,
    "random_walk": random_walk_market,
    "settling": settling_market,
}



def market_definitions(spec: ScenarioSpec) -> dict[str, int]:
    """Close times for the markets a scenario generates.

    Kept separate from the event list on purpose. A close time is part of a
    market's *definition*, not an observation on its feed -- in a real backtest
    it comes from the markets table. Threading it through the tick stream would
    blur a distinction the rest of the platform depends on.
    """
    if spec.close_offset_ms is None:
        return {}
    return {spec.ticker: spec.close_offset_ms}


def generate_scenario(spec: ScenarioSpec) -> list[SimEvent]:
    """Build the history described by ``spec``."""
    generator = _GENERATORS.get(spec.kind)
    if generator is None:
        raise ValueError(
            f"unknown scenario kind {spec.kind!r}; known kinds are "
            f"{sorted(_GENERATORS)}"
        )
    return generator(spec)


def generate_market_series(
    spec: ScenarioSpec,
    *,
    count: int = 12,
    stagger_ms: int | None = None,
) -> tuple[list[SimEvent], dict[str, int]]:
    """A universe of markets that open, trade and settle, one after another.

    A single very long market is the wrong shape for validating an
    event-contract strategy. On Kalshi a position is normally closed by
    *resolution*, not by a later sale, so a run containing one market that never
    resolves produces open positions and no completed trades -- and a backtest
    with no completed trades has measured nothing, however long it ran.

    Overlapping the markets is deliberate too: real trading days have several
    contracts live at once, and a strategy that can only handle one at a time
    should fail here rather than in production.

    Returns ``(events, close_times)``, the second being the market definitions
    :class:`~kat.simulation.features.SimFeatureProvider` needs.
    """
    if count < 1:
        raise ValueError("a market series needs at least one market")
    step = stagger_ms if stagger_ms is not None else spec.duration_ms // 2

    events: list[SimEvent] = []
    closes: dict[str, int] = {}
    for index in range(count):
        offset = step * index
        # Each market gets its own seed so they are independent draws rather
        # than the same path repeated at different times.
        child = replace(
            spec,
            ticker=f"{spec.ticker}-{index:02d}",
            event_ticker=f"{spec.event_ticker}-{index:02d}",
            seed=spec.seed * 1_000 + index,
            settle=True,
        )
        for event in generate_scenario(child):
            events.append(
                replace(
                    event,
                    available_ms=event.available_ms + offset,
                    info_ts_ms=event.info_ts_ms + offset,
                )
            )
        closes[child.ticker] = offset + child.duration_ms
    return events, closes


def realised_volatility_cc(events: list[SimEvent]) -> float:
    """Standard deviation of mid-price changes, for checking a generator.

    A generator asked for volatility 40 that produces 4 is broken, and it is
    much easier to notice that here than to notice it as an oddly good Sharpe
    three layers up.
    """
    mids: list[float] = []
    for event in events:
        bids = event.payload.get("bids")
        asks = event.payload.get("asks")
        if bids and asks:
            mids.append((bids[0][0] + asks[0][0]) / 2.0)
    if len(mids) < 3:
        return 0.0
    diffs = [b - a for a, b in itertools.pairwise(mids)]
    mean = sum(diffs) / len(diffs)
    return math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs))
