"""Market regime detection.

A strategy that works in one regime usually fails in another, and the failure is
rarely dramatic — it is a slow decay that looks like bad luck for weeks. Naming
the regime lets a strategy stand aside from conditions it was never good at,
and lets performance review ask "did it stop working, or did the market change?".

What is measured, and why these
--------------------------------
Four axes, each from a statistic that is cheap and hard to fool:

**Volatility** — realised standard deviation of returns, compared against the
market's *own* recent history rather than an absolute number. A three-cent move
means something different in a market that moves three cents an hour than in one
that moves three cents a minute.

**Liquidity** — spread and depth. This is the axis most often left out, and the
one that most often kills a strategy: an edge in a market that cannot absorb five
contracts is not an edge.

**Trend versus mean reversion** — measured two ways, because one is not enough.

*Efficiency ratio* (net movement divided by total path length) detects
**directional persistence**: a price that walks steadily from 50 to 60 scores
near 1.0 whatever the size of its individual steps.

*Lag-1 autocorrelation of returns* detects **momentum in the changes**: whether a
move up tends to be followed by another move up.

These are genuinely different, and the distinction is not academic. A steady
drift has an efficiency ratio near 1.0 and a return autocorrelation near
**zero**, because its returns are nearly constant rather than correlated. Using
autocorrelation alone — the obvious choice — misses the most obvious trend there
is. Efficiency ratio therefore leads for trend, and negative autocorrelation is
what identifies mean reversion.

**Stress** — volatility high *and* liquidity poor at the same time. Treated as
its own regime rather than as a combination, because it is the condition in which
most strategies should simply stop.

Honest limitations
------------------
These are descriptive statistics over a short window, not a hidden Markov model.
They will label a regime *after* it has begun, not before, and they will be noisy
on thin data — which is why every classification carries a confidence and why
:meth:`RegimeDetector.detect` refuses to classify at all below a minimum sample.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import MarketType
from kat.db.models import RegimeObservation
from kat.db.writer import AsyncWriter
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["Regime", "RegimeDetector", "RegimeReading"]


class Regime:
    """Named regimes. Strings rather than an enum so new ones cost nothing."""

    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    HIGH_LIQUIDITY = "high_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    EVENT_DRIVEN = "event_driven"
    NEWS_DRIVEN = "news_driven"
    STRESSED = "stressed"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RegimeReading:
    """One classification, with the evidence and how much to trust it."""

    primary: str
    confidence: float
    volatility: str = Regime.UNKNOWN
    liquidity: str = Regime.UNKNOWN
    structure: str = Regime.UNKNOWN
    stressed: bool = False
    features: dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    reason: str = ""

    @property
    def all_labels(self) -> list[str]:
        labels = [self.volatility, self.liquidity, self.structure]
        if self.stressed:
            labels.append(Regime.STRESSED)
        return [label for label in labels if label != Regime.UNKNOWN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "confidence": round(self.confidence, 4),
            "volatility": self.volatility,
            "liquidity": self.liquidity,
            "structure": self.structure,
            "stressed": self.stressed,
            "labels": self.all_labels,
            "sample_size": self.sample_size,
            "reason": self.reason,
            "features": {k: round(v, 6) for k, v in self.features.items()},
        }


#: Below this many observations the statistics are noise.
MIN_SAMPLE = 30


def _returns(prices: list[int]) -> list[float]:
    """Simple returns from a centicent price series, skipping non-positive prices."""
    out: list[float] = []
    for previous, current in itertools.pairwise(prices):
        if previous > 0:
            out.append((current - previous) / previous)
    return out


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(0.0, variance))


def _efficiency_ratio(prices: list[int]) -> float:
    """Net movement divided by total path length, in [0, 1].

    Kaufman's efficiency ratio. A price that walks straight from 50 to 60 scores
    1.0; one that ends where it started after wandering scores 0.0. This is what
    detects a *steady drift*, which lag-1 autocorrelation cannot: a constant-step
    trend has near-constant returns, and near-constant returns are uncorrelated.
    """
    if len(prices) < 2:
        return 0.0
    net = abs(prices[-1] - prices[0])
    path = sum(abs(b - a) for a, b in itertools.pairwise(prices))
    if path <= 0:
        return 0.0
    return min(1.0, net / path)


def _autocorrelation(values: list[float], lag: int = 1) -> float:
    """Lag-1 autocorrelation. Positive means trending, negative mean-reverting.

    Returns 0.0 when it cannot be computed, which reads as "no structure" and is
    the right neutral answer rather than a spurious signal in either direction.
    """
    n = len(values)
    if n <= lag + 1:
        return 0.0
    mean = sum(values) / n
    denominator = sum((v - mean) ** 2 for v in values)
    if denominator <= 0:
        return 0.0
    numerator = sum(
        (values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)
    )
    return max(-1.0, min(1.0, numerator / denominator))


class RegimeDetector:
    """Classifies market conditions from a short window of observations."""

    def __init__(
        self,
        *,
        writer: AsyncWriter | None = None,
        high_vol_multiple: float = 1.5,
        low_vol_multiple: float = 0.6,
        trend_threshold: float = 0.15,
        trend_efficiency: float = 0.35,
        wide_spread_cc: int = 300,
        thin_depth_qc: int = 1_000,
    ) -> None:
        self.writer = writer
        #: Volatility is judged against the market's own baseline, not an
        #: absolute number, so these are multiples rather than levels.
        self.high_vol_multiple = high_vol_multiple
        self.low_vol_multiple = low_vol_multiple
        #: Lag-1 autocorrelation beyond this counts as structure rather than noise.
        self.trend_threshold = trend_threshold
        #: Efficiency ratio above this counts as a directional trend. A pure
        #: random walk over n steps scores roughly 1/sqrt(n), so 0.35 is well
        #: clear of noise for the window sizes used here.
        self.trend_efficiency = trend_efficiency
        self.wide_spread_cc = wide_spread_cc
        self.thin_depth_qc = thin_depth_qc

    def detect(
        self,
        *,
        prices_cc: list[int],
        baseline_volatility: float | None = None,
        spread_cc: int | None = None,
        depth_qc: int | None = None,
        news_intensity: float = 0.0,
        minutes_to_close: float | None = None,
    ) -> RegimeReading:
        """Classify from a price window plus current book conditions.

        ``baseline_volatility`` is the market's longer-run volatility. Without it
        the volatility axis is left ``UNKNOWN`` rather than compared against an
        invented constant — a market's own history is the only meaningful
        reference for whether it is currently volatile.
        """
        if len(prices_cc) < MIN_SAMPLE:
            return RegimeReading(
                primary=Regime.UNKNOWN,
                confidence=0.0,
                sample_size=len(prices_cc),
                reason=f"{len(prices_cc)} observations is below the {MIN_SAMPLE} minimum",
            )

        returns = _returns(prices_cc)
        volatility = _stdev(returns)
        autocorrelation = _autocorrelation(returns)
        efficiency = _efficiency_ratio(prices_cc)

        features: dict[str, float] = {
            "realized_volatility": volatility,
            "autocorrelation_lag1": autocorrelation,
            "efficiency_ratio": efficiency,
            "observations": float(len(prices_cc)),
        }

        # --- volatility ----------------------------------------------------
        vol_label = Regime.UNKNOWN
        if baseline_volatility and baseline_volatility > 0:
            ratio = volatility / baseline_volatility
            features["volatility_ratio"] = ratio
            if ratio >= self.high_vol_multiple:
                vol_label = Regime.HIGH_VOLATILITY
            elif ratio <= self.low_vol_multiple:
                vol_label = Regime.LOW_VOLATILITY

        # --- liquidity ------------------------------------------------------
        liquidity_label = Regime.UNKNOWN
        if spread_cc is not None and depth_qc is not None:
            features["spread_cc"] = float(spread_cc)
            features["depth_qc"] = float(depth_qc)
            wide = spread_cc > self.wide_spread_cc
            thin = depth_qc < self.thin_depth_qc
            if wide or thin:
                liquidity_label = Regime.LOW_LIQUIDITY
            elif spread_cc <= self.wide_spread_cc // 2 and depth_qc >= self.thin_depth_qc * 3:
                liquidity_label = Regime.HIGH_LIQUIDITY

        # --- structure ------------------------------------------------------
        # Efficiency ratio leads for trend because it catches a steady drift,
        # which return autocorrelation does not. Negative autocorrelation is
        # what identifies reversion, and it wins when both fire: a series that
        # reverts within a slow drift is more dangerous to trade as a trend.
        if autocorrelation <= -self.trend_threshold:
            structure = Regime.MEAN_REVERTING
        elif efficiency >= self.trend_efficiency or autocorrelation >= self.trend_threshold:
            structure = Regime.TRENDING
        else:
            structure = Regime.UNKNOWN

        # --- driver ---------------------------------------------------------
        # News and imminent resolution both produce volatility, and they call for
        # different responses, so they are distinguished rather than merged.
        features["news_intensity"] = news_intensity
        driver = None
        if news_intensity >= 0.6 and vol_label == Regime.HIGH_VOLATILITY:
            driver = Regime.NEWS_DRIVEN
        elif minutes_to_close is not None and minutes_to_close <= 60:
            features["minutes_to_close"] = minutes_to_close
            driver = Regime.EVENT_DRIVEN

        # --- stress ---------------------------------------------------------
        # Its own regime, not a combination: this is when most strategies should
        # stop rather than adapt.
        stressed = vol_label == Regime.HIGH_VOLATILITY and liquidity_label == Regime.LOW_LIQUIDITY

        primary = (
            Regime.STRESSED
            if stressed
            else driver
            or (structure if structure != Regime.UNKNOWN else None)
            or (vol_label if vol_label != Regime.UNKNOWN else None)
            or (liquidity_label if liquidity_label != Regime.UNKNOWN else Regime.UNKNOWN)
        )

        # Confidence grows with sample size and with how far the statistics sit
        # from their thresholds. A borderline reading is reported as borderline.
        sample_confidence = min(1.0, len(prices_cc) / (MIN_SAMPLE * 4))
        signal_strength = max(
            min(1.0, abs(autocorrelation) / max(self.trend_threshold, 1e-9)),
            min(1.0, efficiency / max(self.trend_efficiency, 1e-9)),
        )
        confidence = round(0.5 * sample_confidence + 0.5 * min(1.0, signal_strength), 4)
        if primary == Regime.UNKNOWN:
            confidence = min(confidence, 0.3)

        return RegimeReading(
            primary=primary,
            confidence=confidence,
            volatility=vol_label,
            liquidity=liquidity_label,
            structure=structure,
            stressed=stressed,
            features=features,
            sample_size=len(prices_cc),
            reason=(
                f"volatility={vol_label}, liquidity={liquidity_label}, "
                f"structure={structure}, autocorrelation={autocorrelation:.3f}, "
                f"efficiency={efficiency:.3f}"
            ),
        )

    def record(
        self,
        reading: RegimeReading,
        *,
        ticker: str | None = None,
        market_type: MarketType | None = None,
    ) -> None:
        """Persist a classification so performance can be reviewed by regime."""
        if self.writer is None:
            return
        self.writer.enqueue(
            RegimeObservation(
                scope="market" if ticker else "global",
                ticker=ticker,
                market_type=market_type.value if market_type else None,
                regime=reading.primary,
                confidence=reading.confidence,
                features=reading.to_dict(),
            )
        )
