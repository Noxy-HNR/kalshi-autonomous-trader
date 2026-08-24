"""Feature definitions, and the timestamps that make leakage *detectable*.

The usual way a research pipeline lies to itself is not a mislabelled column --
it is a feature that was computed from data the strategy could not actually have
held at the moment it traded. A 1-minute bar closing at 12:00:00 is not knowable
at 12:00:00; it is knowable once the bar has closed *and* the exchange has
published it *and* our ingest has stored it. Backtests that ignore that gap
produce Sharpe ratios that evaporate in paper trading, and the failure is silent
because nothing in the data looks wrong.

This module makes the gap explicit and machine-checkable.

Two timestamps, not one
-----------------------
Every feature value carries:

``info_ts_ms``
    The information time: the timestamp of the newest observation that went into
    the value. For a book feature this is the timestamp of the book message; for
    a bar feature it is the close of the newest bar used.

``available_ms``
    ``info_ts_ms + availability_lag_ms`` -- the earliest wall-clock moment at
    which this platform could have held the value. Nothing may act on a feature
    before this instant.

The availability lag is a property of the *pipe*, not of the maths. Book
features carry a lag of zero, because their information timestamp is already the
moment the message was in our hands (see :data:`LAG_BOOK_MS`). Bar features
carry the candle publication delay, because a closed bar reaches us through a
REST poll some seconds later. Getting these numbers slightly wrong is survivable;
not having them at all is not, because then there is nothing to be wrong about.

Two independent defences
------------------------
1. **Construction.** :mod:`kat.features.compute` is handed inputs that have
   already been filtered to what was available at the decision time, so an
   unavailable observation cannot enter a value in the first place.
2. **Detection.** The timestamps travel with the value into
   :class:`FeatureVector`, into the training dataset, and into the ``predictions``
   table. Any row whose ``info_ts_ms`` exceeds its own decision time is provable
   leakage after the fact, months later, without re-running anything.

Belt and braces, because the first defence is code that can be changed by
someone who does not know why it is there.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from kat.common.enums import MarketType
from kat.common.errors import DataQualityError

__all__ = [
    "LAG_BOOK_MS",
    "LAG_CANDLE_MS",
    "LAG_CLOCK_MS",
    "LAG_FUNDING_MS",
    "LAG_TICKER_MS",
    "FeatureFamily",
    "FeatureSetSpec",
    "FeatureSource",
    "FeatureSpec",
    "FeatureValue",
    "FeatureVector",
    "fingerprint_specs",
]


# ---------------------------------------------------------------------------
# Availability lags
# ---------------------------------------------------------------------------
# These describe how long after an observation's information time this platform
# could first hold the derived value. They are deliberately per-source rather
# than per-feature: the delay belongs to the transport, and inventing a
# different number for each feature would be false precision.

#: Order-book features. Zero, because the information timestamp of a book is
#: already the moment the message was applied to local state -- the transport
#: delay is modelled by the timestamp itself, not by an additional lag. Adding a
#: further lag here would double-count it and make live inference refuse to run.
LAG_BOOK_MS: Final[int] = 0

#: Candle features. A bar that closes at ``T`` is fetched by the REST poller
#: afterwards; until then we simply do not have it. Two seconds is the floor we
#: hold ourselves to, and the simulator refuses to use a bar sooner than this.
LAG_CANDLE_MS: Final[int] = 2_000

#: Ticker-derived aggregates (volume, open interest) arrive on the ticker
#: channel with the same publication characteristics as a quote update.
LAG_TICKER_MS: Final[int] = 1_000

#: Funding rate and next-funding time for perpetuals.
LAG_FUNDING_MS: Final[int] = 1_000

#: Features that are pure functions of the platform clock and static market
#: metadata (time to expiry). Known arbitrarily far in advance.
LAG_CLOCK_MS: Final[int] = 0


class FeatureFamily(str, Enum):
    """What kind of market behaviour a feature is trying to capture.

    Used for grouping in the dashboard and, more importantly, for ablation:
    dropping a whole family and retraining is the cheapest test of whether a
    model is leaning on something it should not be.
    """

    MICROSTRUCTURE = "microstructure"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    FLOW = "flow"
    LEVEL = "level"
    TERM = "term"
    CARRY = "carry"


class FeatureSource(str, Enum):
    """Which input a feature is computed from. Determines its availability lag."""

    BOOK = "book"
    CANDLES = "candles"
    TICKER = "ticker"
    FUNDING = "funding"
    CLOCK = "clock"


_SOURCE_LAG_MS: Final[dict[FeatureSource, int]] = {
    FeatureSource.BOOK: LAG_BOOK_MS,
    FeatureSource.CANDLES: LAG_CANDLE_MS,
    FeatureSource.TICKER: LAG_TICKER_MS,
    FeatureSource.FUNDING: LAG_FUNDING_MS,
    FeatureSource.CLOCK: LAG_CLOCK_MS,
}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The definition of one feature.

    ``fill_value`` is the difference between "this feature may be missing" and
    "a missing value is a broken pipeline". A spread cannot be filled -- there is
    no sensible stand-in for an absent book -- so its ``fill_value`` is ``None``
    and its absence aborts the inference. A 60-minute momentum reading during the
    first hour of a market's life is legitimately absent, and filling it with
    zero is both honest and correct.
    """

    name: str
    description: str
    family: FeatureFamily
    source: FeatureSource
    #: Lookback the feature needs, in milliseconds. Zero for point-in-time reads.
    window_ms: int = 0
    #: Which domains the feature is defined for. Funding and basis are perps-only.
    market_types: tuple[MarketType, ...] = (MarketType.EVENT, MarketType.PERP)
    #: ``None`` means required: absence is an error rather than a default.
    fill_value: float | None = None
    clip_low: float | None = None
    clip_high: float | None = None
    #: Overrides the source lag. Set only with a reason, in the spec's description.
    availability_lag_ms_override: int | None = None

    @property
    def availability_lag_ms(self) -> int:
        if self.availability_lag_ms_override is not None:
            return self.availability_lag_ms_override
        return _SOURCE_LAG_MS[self.source]

    @property
    def required(self) -> bool:
        return self.fill_value is None

    def applies_to(self, market_type: MarketType) -> bool:
        return market_type in self.market_types

    def sanitise(self, value: float) -> float:
        """Clip to the declared range and reject non-finite values.

        A NaN reaching a tree model is silently absorbed and reaching a linear
        model poisons the whole prediction. Neither outcome is acceptable on a
        path that ends in an order, so a non-finite value is an error here rather
        than a surprise later.
        """
        if not math.isfinite(value):
            raise DataQualityError(f"feature {self.name!r} computed a non-finite value: {value!r}")
        if self.clip_low is not None:
            value = max(self.clip_low, value)
        if self.clip_high is not None:
            value = min(self.clip_high, value)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "family": self.family.value,
            "source": self.source.value,
            "window_ms": self.window_ms,
            "market_types": [m.value for m in self.market_types],
            "fill_value": self.fill_value,
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
            "availability_lag_ms": self.availability_lag_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSpec:
        source = FeatureSource(payload["source"])
        lag = int(payload.get("availability_lag_ms", _SOURCE_LAG_MS[source]))
        override = lag if lag != _SOURCE_LAG_MS[source] else None
        return cls(
            name=payload["name"],
            description=payload.get("description", ""),
            family=FeatureFamily(payload["family"]),
            source=source,
            window_ms=int(payload.get("window_ms", 0)),
            market_types=tuple(MarketType(m) for m in payload.get("market_types", ["event", "perp"])),
            fill_value=payload.get("fill_value"),
            clip_low=payload.get("clip_low"),
            clip_high=payload.get("clip_high"),
            availability_lag_ms_override=override,
        )


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """One computed value, with the two timestamps that make it auditable."""

    name: str
    value: float
    info_ts_ms: int
    availability_lag_ms: int = 0
    #: True when the value came from ``FeatureSpec.fill_value`` rather than data.
    imputed: bool = False

    @property
    def available_ms(self) -> int:
        return self.info_ts_ms + self.availability_lag_ms

    def is_available_at(self, now_ms: int) -> bool:
        return self.available_ms <= now_ms


@dataclass(slots=True)
class FeatureVector:
    """The features for one market at one decision time.

    The same object is produced by the training pipeline and by the live
    inference path -- that identity is the point of the whole subsystem. If a
    backtest and a live decision can disagree about what the features were, then
    the backtest was measuring a different strategy.
    """

    ticker: str
    market_type: MarketType
    as_of_ms: int
    feature_set_id: str
    values: dict[str, FeatureValue] = field(default_factory=dict)
    #: Names that fell back to their declared ``fill_value``.
    imputed: tuple[str, ...] = ()

    # -- access --------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.values)

    def get(self, name: str, default: float = 0.0) -> float:
        fv = self.values.get(name)
        return fv.value if fv is not None else default

    def as_dict(self) -> dict[str, float]:
        return {name: fv.value for name, fv in self.values.items()}

    def ordered(self, names: tuple[str, ...] | list[str]) -> list[float]:
        """Values in the exact order the model was trained on.

        Column order is part of a model's contract. A silently reordered vector
        is one of the few bugs that produces plausible-looking predictions
        indefinitely, so a missing name is raised rather than defaulted.
        """
        out: list[float] = []
        missing: list[str] = []
        for name in names:
            fv = self.values.get(name)
            if fv is None:
                missing.append(name)
            else:
                out.append(fv.value)
        if missing:
            raise DataQualityError(
                f"feature vector for {self.ticker} is missing {len(missing)} required "
                f"feature(s): {', '.join(sorted(missing))}"
            )
        return out

    # -- timestamps ----------------------------------------------------------

    @property
    def max_info_ts_ms(self) -> int:
        """Information time of the newest observation in the vector.

        This is what gets written to ``predictions.feature_ts``.
        """
        return max((fv.info_ts_ms for fv in self.values.values()), default=self.as_of_ms)

    @property
    def max_available_ms(self) -> int:
        return max((fv.available_ms for fv in self.values.values()), default=self.as_of_ms)

    def unavailable_at(self, now_ms: int) -> tuple[str, ...]:
        """Features that could not yet have been computed at ``now_ms``."""
        return tuple(sorted(n for n, fv in self.values.items() if not fv.is_available_at(now_ms)))

    def leaking(self) -> tuple[str, ...]:
        """Features whose availability postdates this vector's own decision time.

        Non-empty means the vector saw the future. There is no severity scale
        here: any entry is a bug in the pipeline that produced it.
        """
        return self.unavailable_at(self.as_of_ms)

    def require_point_in_time(self) -> None:
        offenders = self.leaking()
        if offenders:
            details = ", ".join(
                f"{n}(avail={self.values[n].available_ms} > as_of={self.as_of_ms})" for n in offenders
            )
            raise DataQualityError(f"look-ahead in feature vector for {self.ticker}: {details}")

    # -- serialisation -------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """JSON payload for ``predictions.features``.

        Stores the per-feature information times alongside the values. Storing
        only the values would make a prediction reproducible but not auditable:
        we could recompute the number without ever being able to prove it was
        computable at the time.
        """
        return {
            "as_of_ms": self.as_of_ms,
            "feature_set_id": self.feature_set_id,
            "values": self.as_dict(),
            "info_ts_ms": {n: fv.info_ts_ms for n, fv in self.values.items()},
            "imputed": list(self.imputed),
        }


def fingerprint_specs(specs: tuple[FeatureSpec, ...]) -> str:
    """Content hash of a set of feature definitions.

    Used in the feature-set id so that editing a definition without bumping the
    version produces a different id rather than quietly invalidating every model
    trained against the old meaning of the same name.
    """
    payload = json.dumps([s.to_dict() for s in specs], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureSetSpec:
    """A named, versioned, ordered set of feature definitions.

    Order is significant and preserved: it is the column order of every training
    matrix and every inference vector built from this set.
    """

    name: str
    version: str
    market_type: MarketType
    specs: tuple[FeatureSpec, ...]
    description: str = ""
    git_commit: str | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for spec in self.specs:
            if spec.name in seen:
                raise DataQualityError(f"duplicate feature name in feature set {self.name}: {spec.name}")
            seen.add(spec.name)
            if not spec.applies_to(self.market_type):
                raise DataQualityError(
                    f"feature {spec.name!r} is not defined for market type "
                    f"{self.market_type.value} but is listed in feature set {self.name}"
                )

    @property
    def fingerprint(self) -> str:
        return fingerprint_specs(self.specs)

    @property
    def feature_set_id(self) -> str:
        """Stable identity: name, version, and a hash of the actual definitions."""
        return f"{self.name}.{self.version}.{self.fingerprint[:12]}"

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    @property
    def availability_lag_ms(self) -> dict[str, int]:
        """Per-feature lag map, as persisted to ``feature_sets.availability_lag_ms``."""
        return {s.name: s.availability_lag_ms for s in self.specs}

    @property
    def max_window_ms(self) -> int:
        """Warm-up requirement: no vector from this set is valid before this much history."""
        return max((s.window_ms for s in self.specs), default=0)

    def get(self, name: str) -> FeatureSpec | None:
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

    def subset(self, names: tuple[str, ...] | list[str]) -> FeatureSetSpec:
        """A narrower set, e.g. after feature selection dropped columns."""
        keep = set(names)
        chosen = tuple(s for s in self.specs if s.name in keep)
        missing = keep - {s.name for s in chosen}
        if missing:
            raise DataQualityError(f"cannot subset to unknown features: {sorted(missing)}")
        return FeatureSetSpec(
            name=self.name,
            version=self.version,
            market_type=self.market_type,
            specs=chosen,
            description=self.description,
            git_commit=self.git_commit,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set_id": self.feature_set_id,
            "name": self.name,
            "version": self.version,
            "market_type": self.market_type.value,
            "description": self.description,
            "git_commit": self.git_commit,
            "fingerprint": self.fingerprint,
            "features": [s.to_dict() for s in self.specs],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSetSpec:
        return cls(
            name=payload["name"],
            version=payload["version"],
            market_type=MarketType(payload["market_type"]),
            specs=tuple(FeatureSpec.from_dict(f) for f in payload.get("features", [])),
            description=payload.get("description", ""),
            git_commit=payload.get("git_commit"),
        )
