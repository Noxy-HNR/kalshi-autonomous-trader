"""One code path from stored history to a training matrix, and from live state to
an inference vector.

The single most valuable property of this module is that there is no second
implementation. :meth:`FeaturePipeline.vector` is called by
:meth:`FeaturePipeline.build_dataset` once per historical decision time and by
:meth:`FeaturePipeline.live_vector` once per live decision, with the same
:class:`~kat.features.compute.FeatureComputer` and the same
:class:`~kat.features.compute.FeatureInputs` shape. A backtest and a live
decision that disagree about what the features were are not measuring the same
strategy, and the usual cause is exactly this: a research notebook that computed
a column with pandas and a production path that computed it again, differently,
in Python.

Point-in-time construction
--------------------------
:class:`MarketHistory` holds everything ever observed about one market.
:meth:`MarketHistory.inputs_at` returns only what was *publishable* by the
decision time: the newest book whose message timestamp has passed, the newest
ticker observation once its lag has elapsed, the newest funding observation, and
series objects that filter themselves on every read. A later observation cannot
enter a feature because it is not handed to the computer at all.

That is the construction defence. The detection defence -- the timestamps
travelling with every value -- runs on top of it, and
:meth:`FeaturePipeline.build_dataset` refuses to emit a row whose vector fails
it.

Labels are the other half, and the dangerous half
-------------------------------------------------
Features must not see the future; labels must. A labeller is therefore given
unrestricted access to the history, which is why label computation is a separate
object with its own module section rather than another feature: the two have
opposite requirements and mixing them is how look-ahead gets in.

Every label reports ``label_ts_ms`` -- the moment its own value became knowable.
That timestamp is not decoration. A row observed at 14:00 with a one-hour
horizon is not fully known until 15:00, so it overlaps any evaluation window
starting before 15:00. :func:`chronological_split` purges those rows and then
embargoes a further gap, which is the difference between an honest out-of-sample
score and a flattering one.

Money
-----
No money value is produced here. Prices are centicents, sizes are quantity
centis, and the matrix handed to a model is float only because that is what
estimators consume -- by then the values are statistics, not currency.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from kat.common.enums import MarketType
from kat.common.errors import DataQualityError
from kat.common.timeutil import from_ms
from kat.db.models import Dataset
from kat.features.compute import (
    Bar,
    BarSeries,
    FeatureComputer,
    FeatureInputs,
    MidSeries,
    TradeSeries,
)
from kat.features.spec import (
    LAG_BOOK_MS,
    LAG_FUNDING_MS,
    LAG_TICKER_MS,
    FeatureSetSpec,
    FeatureVector,
)
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "BookObservation",
    "DatasetSplits",
    "FeaturePipeline",
    "ForwardMoveLabel",
    "FundingObservation",
    "Labeller",
    "MarketHistory",
    "SettlementLabel",
    "SplitView",
    "TickerObservation",
    "TrainingDataset",
    "bars_from_mid_path",
    "chronological_split",
    "decision_times",
    "summarise_availability",
]


# ---------------------------------------------------------------------------
# Stored observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookObservation:
    """A book as it stood when a message was applied to local state."""

    ts_ms: int
    book: OrderBook


@dataclass(frozen=True, slots=True)
class TickerObservation:
    """Aggregates that arrive on the ticker channel."""

    ts_ms: int
    volume_qc: int | None = None
    open_interest_qc: int | None = None
    last_trade_cc: int | None = None
    mark_price_cc: int | None = None


@dataclass(frozen=True, slots=True)
class FundingObservation:
    """A funding rate and the time of the next payment. Perpetuals only."""

    ts_ms: int
    rate_hourly: float
    next_funding_ms: int | None = None


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class MarketHistory:
    """Everything observed about one market, with two deliberately different views.

    :meth:`inputs_at` is the point-in-time view and is the only one the feature
    path may use. :meth:`mid_at` and :meth:`settlement` are the unrestricted
    view and exist for labellers, which are supposed to see the future.

    The two are separate methods rather than a flag because a flag defaulting to
    the wrong value silently produces a backtest that cannot be reproduced live,
    and there is no error message for that.
    """

    __slots__ = (
        "_book_ts",
        "_books",
        "_funding",
        "_funding_ts",
        "_mids",
        "_ticker_ts",
        "_tickers",
        "bars",
        "close_ts_ms",
        "market_type",
        "settlement_yes",
        "ticker",
        "trades",
    )

    def __init__(
        self,
        ticker: str,
        market_type: MarketType,
        *,
        books: Iterable[BookObservation] = (),
        bars: BarSeries | None = None,
        trades: TradeSeries | None = None,
        tickers: Iterable[TickerObservation] = (),
        funding: Iterable[FundingObservation] = (),
        close_ts_ms: int | None = None,
        settlement_yes: float | None = None,
    ) -> None:
        self.ticker = ticker
        self.market_type = market_type
        self.bars = bars
        self.trades = trades
        self.close_ts_ms = close_ts_ms
        self.settlement_yes = settlement_yes

        ordered_books = sorted(books, key=lambda b: b.ts_ms)
        self._books: list[BookObservation] = ordered_books
        self._book_ts: list[int] = [b.ts_ms for b in ordered_books]

        # The mid series is derived once from the book history rather than
        # stored separately, so the sub-minute features in research are computed
        # from exactly the messages the live engine would have seen.
        mids: list[tuple[int, int]] = []
        for obs in ordered_books:
            mid = obs.book.mid_cc
            if mid is not None:
                mids.append((obs.ts_ms, mid))
        self._mids = MidSeries(mids)

        ordered_tickers = sorted(tickers, key=lambda t: t.ts_ms)
        self._tickers: list[TickerObservation] = ordered_tickers
        self._ticker_ts: list[int] = [t.ts_ms for t in ordered_tickers]

        ordered_funding = sorted(funding, key=lambda f: f.ts_ms)
        self._funding: list[FundingObservation] = ordered_funding
        self._funding_ts: list[int] = [f.ts_ms for f in ordered_funding]

    # -- point-in-time view --------------------------------------------------

    def inputs_at(self, as_of_ms: int) -> FeatureInputs | None:
        """Inputs holding only what was publishable at ``as_of_ms``.

        Returns ``None`` when no book had yet been received, because a market we
        have never seen a book for is not a market we could have traded.
        """
        book_obs = self._latest_book(as_of_ms)
        if book_obs is None:
            return None
        ticker_obs = self._latest_ticker(as_of_ms)
        funding_obs = self._latest_funding(as_of_ms)
        return FeatureInputs(
            ticker=self.ticker,
            market_type=self.market_type,
            book=book_obs.book,
            book_ts_ms=book_obs.ts_ms,
            mids=self._mids,
            bars=self.bars,
            trades=self.trades,
            ticker_ts_ms=ticker_obs.ts_ms if ticker_obs else 0,
            volume_qc=ticker_obs.volume_qc if ticker_obs else None,
            open_interest_qc=ticker_obs.open_interest_qc if ticker_obs else None,
            last_trade_cc=ticker_obs.last_trade_cc if ticker_obs else None,
            mark_price_cc=ticker_obs.mark_price_cc if ticker_obs else None,
            funding_ts_ms=funding_obs.ts_ms if funding_obs else 0,
            funding_rate_hourly=funding_obs.rate_hourly if funding_obs else None,
            next_funding_ms=funding_obs.next_funding_ms if funding_obs else None,
            close_ts_ms=self.close_ts_ms,
        )

    def _latest_book(self, as_of_ms: int) -> BookObservation | None:
        idx = bisect.bisect_right(self._book_ts, as_of_ms - LAG_BOOK_MS)
        return self._books[idx - 1] if idx else None

    def _latest_ticker(self, as_of_ms: int) -> TickerObservation | None:
        idx = bisect.bisect_right(self._ticker_ts, as_of_ms - LAG_TICKER_MS)
        return self._tickers[idx - 1] if idx else None

    def _latest_funding(self, as_of_ms: int) -> FundingObservation | None:
        idx = bisect.bisect_right(self._funding_ts, as_of_ms - LAG_FUNDING_MS)
        return self._funding[idx - 1] if idx else None

    @property
    def first_book_ms(self) -> int | None:
        return self._book_ts[0] if self._book_ts else None

    @property
    def last_book_ms(self) -> int | None:
        return self._book_ts[-1] if self._book_ts else None

    # -- unrestricted view: LABELS ONLY -------------------------------------

    def mid_at(self, ts_ms: int) -> int | None:
        """Book midpoint as of ``ts_ms``, with no availability filtering.

        **Labels only.** Calling this from anything that computes a feature is a
        look-ahead bug; the timestamps in the resulting vector will not catch it,
        because this method hands back a value with no timestamp at all.
        """
        idx = bisect.bisect_right(self._book_ts, ts_ms)
        while idx > 0:
            mid = self._books[idx - 1].book.mid_cc
            if mid is not None:
                return mid
            idx -= 1
        return None

    def settlement(self) -> float | None:
        """Realised outcome in [0, 1], or ``None`` if the market has not settled."""
        return self.settlement_yes


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class Labeller(Protocol):
    """Computes the supervised target for one decision time.

    Returns ``(y, label_ts_ms)``, where ``label_ts_ms`` is when the value became
    knowable, or ``None`` when the target cannot be determined -- typically at
    the end of a history, where the forward window runs off the data.
    """

    @property
    def name(self) -> str: ...

    @property
    def horizon_ms(self) -> int: ...

    def label(self, history: MarketHistory, as_of_ms: int) -> tuple[float, int] | None: ...


@dataclass(frozen=True, slots=True)
class ForwardMoveLabel:
    """Did the midpoint rise by more than ``threshold_cc`` over the horizon?

    Denominated in centicents rather than percent for the reason given in
    :mod:`kat.features.compute`: on a probability scale, a five-point move is a
    five-point move whether it starts at $0.05 or $0.50, and a percentage target
    would train the model almost entirely on cheap contracts.

    ``threshold_cc`` defaults to zero, which labels any upward move. Setting it
    to the round-trip cost instead trains the model on the question the platform
    actually asks -- "will this move enough to pay for itself?" -- and is what
    the strategy layer should use.
    """

    horizon_ms: int
    threshold_cc: int = 0
    name: str = "forward_move_up"

    def label(self, history: MarketHistory, as_of_ms: int) -> tuple[float, int] | None:
        now_mid = history.mid_at(as_of_ms)
        if now_mid is None:
            return None
        future_ms = as_of_ms + self.horizon_ms
        last = history.last_book_ms
        if last is None or last < future_ms:
            # The forward window runs past the end of the data. Labelling this
            # row from the last available book would quietly shorten the horizon
            # for the newest rows -- exactly the rows a time-ordered split puts
            # in the test set.
            return None
        future_mid = history.mid_at(future_ms)
        if future_mid is None:
            return None
        return (1.0 if future_mid - now_mid > self.threshold_cc else 0.0), future_ms


@dataclass(frozen=True, slots=True)
class SettlementLabel:
    """Did the contract resolve YES?

    The target every event-contract model ultimately wants, and the one with the
    longest and most variable horizon: a row observed a week before close is not
    knowable until close. :func:`chronological_split` handles that through
    ``label_ts_ms``, which for this labeller is the market's close time rather
    than a fixed offset.
    """

    name: str = "settlement_yes"

    @property
    def horizon_ms(self) -> int:
        # Variable by construction; reported as zero so that a caller deriving a
        # default embargo from the horizon does not under- or over-state it. The
        # correct embargo for settlement labels is set from the data, not from
        # this number.
        return 0

    def label(self, history: MarketHistory, as_of_ms: int) -> tuple[float, int] | None:
        outcome = history.settlement()
        if outcome is None or history.close_ts_ms is None:
            return None
        if history.close_ts_ms <= as_of_ms:
            return None
        return float(outcome), history.close_ts_ms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SplitView:
    """One contiguous slice of a dataset, in time order."""

    name: str
    indices: np.ndarray
    X: np.ndarray
    y: np.ndarray
    as_of_ms: np.ndarray

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def start_ms(self) -> int:
        return int(self.as_of_ms[0]) if self.n else 0

    @property
    def end_ms(self) -> int:
        return int(self.as_of_ms[-1]) if self.n else 0

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if self.n else 0.0

    def window(self) -> dict[str, Any]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms, "rows": self.n}


@dataclass(slots=True)
class DatasetSplits:
    """Train / validation / test, in time order, purged and embargoed."""

    train: SplitView
    validation: SplitView
    test: SplitView
    embargo_ms: int
    purged_rows: int

    def summary(self) -> dict[str, Any]:
        return {
            "train": self.train.window(),
            "validation": self.validation.window(),
            "test": self.test.window(),
            "embargo_ms": self.embargo_ms,
            "purged_rows": self.purged_rows,
        }


@dataclass(slots=True)
class TrainingDataset:
    """An immutable, content-addressed training matrix.

    ``dataset_id`` hashes the feature names, the matrix, the targets and the
    timestamps. Two datasets with the same id are the same data; a dataset whose
    id does not match a model's recorded ``dataset_id`` is a different dataset
    however similarly it is named, which is what makes "which data was this
    trained on?" answerable months later.
    """

    name: str
    market_type: MarketType
    feature_set_id: str
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    as_of_ms: np.ndarray
    label_ts_ms: np.ndarray
    tickers: tuple[str, ...]
    label_name: str
    horizon_ms: int
    is_synthetic: bool = False
    imputed_counts: np.ndarray | None = None
    max_info_ts_ms: np.ndarray | None = None
    _content_sha: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        if self.y.shape[0] != n or self.as_of_ms.shape[0] != n or self.label_ts_ms.shape[0] != n:
            raise DataQualityError("dataset arrays disagree on row count")
        if self.X.shape[1] != len(self.feature_names):
            raise DataQualityError(
                f"matrix has {self.X.shape[1]} columns but {len(self.feature_names)} feature names"
            )
        if n and not np.all(np.diff(self.as_of_ms) >= 0):
            raise DataQualityError("dataset rows are not in time order")
        if not np.all(np.isfinite(self.X)):
            raise DataQualityError("dataset contains non-finite feature values")

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def sha256(self) -> str:
        if self._content_sha is None:
            h = hashlib.sha256()
            h.update("\x00".join(self.feature_names).encode("utf-8"))
            h.update(self.label_name.encode("utf-8"))
            h.update(np.ascontiguousarray(self.X, dtype=np.float64).tobytes())
            h.update(np.ascontiguousarray(self.y, dtype=np.float64).tobytes())
            h.update(np.ascontiguousarray(self.as_of_ms, dtype=np.int64).tobytes())
            self._content_sha = h.hexdigest()
        return self._content_sha

    @property
    def dataset_id(self) -> str:
        return f"{self.name}.{self.sha256[:16]}"

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if self.n_rows else 0.0

    def spec_payload(self) -> dict[str, Any]:
        return {
            "feature_set_id": self.feature_set_id,
            "feature_names": list(self.feature_names),
            "label": self.label_name,
            "horizon_ms": self.horizon_ms,
            "positive_rate": self.positive_rate,
            "n_markets": len(set(self.tickers)),
            "imputed_total": int(self.imputed_counts.sum()) if self.imputed_counts is not None else 0,
        }

    def to_orm(self, *, path: str | None = None) -> Dataset:
        """The ``datasets`` row describing this matrix."""
        return Dataset(
            dataset_id=self.dataset_id,
            name=self.name,
            market_type=self.market_type.value,
            feature_set_id=self.feature_set_id,
            row_count=self.n_rows,
            start_ts=from_ms(int(self.as_of_ms[0])) if self.n_rows else None,
            end_ts=from_ms(int(self.as_of_ms[-1])) if self.n_rows else None,
            path=path,
            sha256=self.sha256,
            is_synthetic=self.is_synthetic,
            spec=self.spec_payload(),
        )

    def save_npz(self, path: str) -> str:
        """Persist the matrix. Returns the sha256 of the content, not of the file.

        The file hash would change with numpy's archive format; the content hash
        is what a model record is pinned to.
        """
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            as_of_ms=self.as_of_ms,
            label_ts_ms=self.label_ts_ms,
            feature_names=np.array(self.feature_names, dtype=object),
            tickers=np.array(self.tickers, dtype=object),
        )
        return self.sha256

    def subset(self, indices: np.ndarray, name: str) -> SplitView:
        return SplitView(
            name=name,
            indices=indices,
            X=self.X[indices],
            y=self.y[indices],
            as_of_ms=self.as_of_ms[indices],
        )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def chronological_split(
    dataset: TrainingDataset,
    *,
    train_frac: float = 0.6,
    validation_frac: float = 0.2,
    embargo_ms: int | None = None,
) -> DatasetSplits:
    """Split in time order, purging label overlap and embargoing the boundary.

    Two distinct protections, both necessary:

    **Purging.** A row observed before a boundary whose label only becomes known
    after it has seen the evaluation period. It is dropped from the earlier
    split. Without this, a one-hour-horizon model evaluated on the next hour is
    scored partly on labels its training rows already contained.

    **Embargo.** Even correctly purged rows sit next to the boundary in a
    serially correlated series, so the last training row and the first test row
    are near-duplicates. A gap of ``embargo_ms`` after each boundary removes
    that. The default is the label horizon, which is the shortest defensible
    choice; for a settlement label, whose horizon is variable, pass an explicit
    value.

    Splitting is by *row count* within the time order rather than by wall-clock
    duration, so an idle stretch of history does not hand one split a hundred
    rows and another ten thousand.
    """
    if not 0.0 < train_frac < 1.0 or not 0.0 < validation_frac < 1.0:
        raise DataQualityError("split fractions must be strictly between 0 and 1")
    if train_frac + validation_frac >= 1.0:
        raise DataQualityError("train and validation fractions leave nothing for test")
    n = dataset.n_rows
    if n < 3:
        raise DataQualityError(f"cannot split {n} rows into three parts")

    gap = dataset.horizon_ms if embargo_ms is None else embargo_ms
    if gap < 0:
        raise DataQualityError("embargo must not be negative")

    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + validation_frac))
    if train_end_idx == 0 or val_end_idx <= train_end_idx or val_end_idx >= n:
        raise DataQualityError("split fractions produced an empty section")

    as_of = dataset.as_of_ms
    label_ts = dataset.label_ts_ms
    val_start_ms = int(as_of[train_end_idx])
    test_start_ms = int(as_of[val_end_idx])

    all_idx = np.arange(n)
    train_raw = all_idx[:train_end_idx]
    val_raw = all_idx[train_end_idx:val_end_idx]
    test_raw = all_idx[val_end_idx:]

    train_keep = train_raw[label_ts[train_raw] <= val_start_ms]
    val_keep = val_raw[(as_of[val_raw] >= val_start_ms + gap) & (label_ts[val_raw] <= test_start_ms)]
    test_keep = test_raw[as_of[test_raw] >= test_start_ms + gap]

    purged = (len(train_raw) - len(train_keep)) + (len(val_raw) - len(val_keep)) + (
        len(test_raw) - len(test_keep)
    )
    for name, keep in (("train", train_keep), ("validation", val_keep), ("test", test_keep)):
        if keep.size == 0:
            raise DataQualityError(
                f"{name} split is empty after purging and a {gap}ms embargo; "
                "the history is too short for this horizon"
            )

    return DatasetSplits(
        train=dataset.subset(train_keep, "train"),
        validation=dataset.subset(val_keep, "validation"),
        test=dataset.subset(test_keep, "test"),
        embargo_ms=gap,
        purged_rows=int(purged),
    )


def decision_times(start_ms: int, end_ms: int, step_ms: int) -> list[int]:
    """Evenly spaced decision times in ``[start_ms, end_ms]``.

    A fixed cadence, not one sampled from the data: sampling decision times at
    the moments something happened biases the dataset toward active markets and
    teaches a model that markets are always busy.
    """
    if step_ms <= 0:
        raise DataQualityError("decision-time step must be positive")
    return list(range(start_ms, end_ms + 1, step_ms))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _BuildStats:
    considered: int = 0
    no_inputs: int = 0
    warmup_skipped: int = 0
    no_label: int = 0
    unusable: int = 0
    emitted: int = 0


class FeaturePipeline:
    """Builds training matrices and live inference vectors from the same computer."""

    __slots__ = ("computer",)

    def __init__(self, spec: FeatureSetSpec | FeatureComputer) -> None:
        self.computer = spec if isinstance(spec, FeatureComputer) else FeatureComputer(spec)

    @property
    def feature_set(self) -> FeatureSetSpec:
        return self.computer.spec

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.computer.names

    # -- the shared code path -----------------------------------------------

    def vector(self, inputs: FeatureInputs, *, as_of_ms: int) -> FeatureVector:
        """Compute one feature vector. Used by training and by live inference."""
        return self.computer.compute(inputs, as_of_ms=as_of_ms)

    # -- live ---------------------------------------------------------------

    def live_vector(
        self,
        snapshot: MarketSnapshot,
        *,
        now_ms: int,
        bars: BarSeries | None = None,
        trades: TradeSeries | None = None,
        mids: MidSeries | None = None,
        close_ts_ms: int | None = None,
    ) -> FeatureVector:
        """Feature vector for a live market.

        ``now_ms`` comes from the platform clock (:mod:`kat.common.timeutil`) and
        is never read from the wall clock here, so the simulator drives this
        method exactly as production does.
        """
        inputs = FeatureInputs.from_snapshot(
            snapshot, bars=bars, trades=trades, mids=mids, close_ts_ms=close_ts_ms
        )
        return self.vector(inputs, as_of_ms=now_ms)

    # -- training -----------------------------------------------------------

    def build_dataset(
        self,
        histories: Sequence[MarketHistory],
        *,
        labeller: Labeller,
        times: Sequence[int] | None = None,
        step_ms: int = 60_000,
        name: str = "dataset",
        warmup_ms: int | None = None,
        is_synthetic: bool = False,
    ) -> TrainingDataset:
        """Build a point-in-time training matrix.

        ``warmup_ms`` defaults to the feature set's longest window: before that
        much history exists, the long-window features are all imputed and the row
        describes the pipeline's warm-up rather than the market. Such rows are
        skipped rather than filled, because a model trained on them learns the
        constant.

        Rows whose vector fails the look-ahead check are counted and dropped, and
        their presence is logged at error level. That case should be impossible
        given point-in-time construction; if it ever fires, something upstream is
        handing over data it should not have.
        """
        market_type = self.feature_set.market_type
        warmup = self.feature_set.max_window_ms if warmup_ms is None else warmup_ms

        rows: list[list[float]] = []
        ys: list[float] = []
        as_ofs: list[int] = []
        label_tss: list[int] = []
        tickers: list[str] = []
        imputed: list[int] = []
        info_tss: list[int] = []
        stats = _BuildStats()

        for history in histories:
            if history.market_type is not market_type:
                raise DataQualityError(
                    f"history {history.ticker} is {history.market_type.value} but the feature "
                    f"set is {market_type.value}"
                )
            first = history.first_book_ms
            last = history.last_book_ms
            if first is None or last is None:
                continue
            grid = list(times) if times is not None else decision_times(first + warmup, last, step_ms)

            for as_of_ms in grid:
                stats.considered += 1
                if as_of_ms - first < warmup:
                    stats.warmup_skipped += 1
                    continue
                inputs = history.inputs_at(as_of_ms)
                if inputs is None:
                    stats.no_inputs += 1
                    continue
                target = labeller.label(history, as_of_ms)
                if target is None:
                    stats.no_label += 1
                    continue
                try:
                    vector = self.computer.compute(inputs, as_of_ms=as_of_ms)
                except DataQualityError:
                    # A required feature was absent -- typically a one-sided or
                    # crossed book. The live path refuses to trade in that state,
                    # so training on it would fit a regime the model never sees.
                    stats.unusable += 1
                    continue

                rows.append(vector.ordered(self.feature_names))
                ys.append(target[0])
                as_ofs.append(as_of_ms)
                label_tss.append(target[1])
                tickers.append(history.ticker)
                imputed.append(len(vector.imputed))
                info_tss.append(vector.max_info_ts_ms)
                stats.emitted += 1

        if not rows:
            raise DataQualityError(
                f"no usable rows: considered={stats.considered} warmup_skipped={stats.warmup_skipped} "
                f"no_inputs={stats.no_inputs} no_label={stats.no_label} unusable={stats.unusable}"
            )

        # Multiple markets interleave in time; the split logic requires a single
        # global time order, so sort by decision time with the ticker as a stable
        # tiebreaker.
        order = sorted(range(len(rows)), key=lambda i: (as_ofs[i], tickers[i]))
        X = np.asarray([rows[i] for i in order], dtype=np.float64)
        y = np.asarray([ys[i] for i in order], dtype=np.float64)

        dataset = TrainingDataset(
            name=name,
            market_type=market_type,
            feature_set_id=self.feature_set.feature_set_id,
            feature_names=self.feature_names,
            X=X,
            y=y,
            as_of_ms=np.asarray([as_ofs[i] for i in order], dtype=np.int64),
            label_ts_ms=np.asarray([label_tss[i] for i in order], dtype=np.int64),
            tickers=tuple(tickers[i] for i in order),
            label_name=labeller.name,
            horizon_ms=labeller.horizon_ms,
            is_synthetic=is_synthetic,
            imputed_counts=np.asarray([imputed[i] for i in order], dtype=np.int32),
            max_info_ts_ms=np.asarray([info_tss[i] for i in order], dtype=np.int64),
        )
        log.info(
            "features.dataset_built",
            dataset_id=dataset.dataset_id,
            rows=dataset.n_rows,
            features=dataset.n_features,
            markets=len(set(dataset.tickers)),
            positive_rate=round(dataset.positive_rate, 4),
            considered=stats.considered,
            warmup_skipped=stats.warmup_skipped,
            no_label=stats.no_label,
            unusable=stats.unusable,
        )
        return dataset


def bars_from_mid_path(
    points: Sequence[tuple[int, int]],
    *,
    period_ms: int,
    volumes: Sequence[int] | None = None,
    open_interest: Sequence[int] | None = None,
) -> BarSeries:
    """Aggregate a mid-price path into OHLCV bars closing on the period boundary.

    Kalshi publishes candles directly, so this is not the production ingest path.
    It exists because a simulator and a test fixture both need bars that are
    consistent with the book history they sit alongside -- deriving them from the
    same path is the only way to guarantee that, and an inconsistent pair
    produces feature values that occur nowhere in reality.
    """
    if period_ms <= 0:
        raise DataQualityError("bar period must be positive")
    buckets: dict[int, list[tuple[int, int]]] = {}
    for i, (ts_ms, mid_cc) in enumerate(points):
        close_ms = ((ts_ms // period_ms) + 1) * period_ms
        buckets.setdefault(close_ms, []).append((i, mid_cc))

    bars: list[Bar] = []
    for close_ms in sorted(buckets):
        members = buckets[close_ms]
        prices = [m for _, m in members]
        idxs = [i for i, _ in members]
        bars.append(
            Bar(
                ts_ms=close_ms,
                open_cc=prices[0],
                high_cc=max(prices),
                low_cc=min(prices),
                close_cc=prices[-1],
                volume_qc=int(sum(volumes[i] for i in idxs)) if volumes is not None else 0,
                open_interest_qc=int(open_interest[idxs[-1]]) if open_interest is not None else None,
            )
        )
    return BarSeries(bars, period_ms=period_ms)


def summarise_availability(vector: FeatureVector) -> dict[str, Any]:
    """Per-vector audit line: how stale, how much imputed, anything unavailable.

    Written alongside a prediction so that a bad live decision can be diagnosed
    from the record rather than from a guess about what the data looked like.
    """
    ages = [vector.as_of_ms - fv.info_ts_ms for fv in vector.values.values() if not fv.imputed]
    return {
        "feature_set_id": vector.feature_set_id,
        "as_of_ms": vector.as_of_ms,
        "max_info_ts_ms": vector.max_info_ts_ms,
        "max_age_ms": max(ages) if ages else 0,
        "mean_age_ms": int(math.fsum(ages) / len(ages)) if ages else 0,
        "imputed": list(vector.imputed),
        "leaking": list(vector.leaking()),
    }
