"""Train a model from banked quotes, and deploy it if it earns deployment.

Every piece of this existed already -- labellers, a point-in-time dataset
builder, four trainer families, calibration, selection gates, a registry with
deployment and rollback -- and nothing drove them. So no model was ever
deployed, and ``no_model_probability`` was the single largest reason the
platform declined to trade.

What this does, in order:

1. Rebuilds one :class:`MarketHistory` per market from the ``quotes`` table.
2. Labels each decision time with :class:`ForwardMoveLabel`. The threshold is a
   real distance rather than zero: "did it move at all" is not the question a
   strategy that pays a spread and a fee is asking.
3. Builds the matrix through the *same* ``FeaturePipeline`` the live worker
   uses, so a number here and a number in production are the same number.
4. Splits by time, never at random, and trains every family on identical data.
5. Hands the result to :func:`select_best`, which applies the platform's own
   hard gates and is entitled to return nothing.
6. Registers the winner, and deploys it only when asked.

Usage::

    .venv/Scripts/python.exe scripts/train_model.py
    .venv/Scripts/python.exe scripts/train_model.py --deploy

Not deploying is the default, because deploying changes what a running platform
does with money.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from kat.common.enums import MarketType
from kat.config.settings import AppSettings
from kat.db.engine import get_database
from kat.features.compute import EVENT_FEATURE_SET
from kat.features.pipeline import (
    BookObservation,
    FeaturePipeline,
    ForwardMoveLabel,
    MarketHistory,
    SettlementLabel,
    TickerObservation,
    bars_from_mid_path,
)
from kat.marketdata.book import OrderBook
from kat.models.registry import ModelRegistry
from kat.models.selection import select_best
from kat.models.trainers import temporal_split, train_all
from kat.observability.logging import configure_logging, get_logger

log = get_logger("train")

#: A market needs enough distinct prices for a move label to carry information.
#: One that never repriced contributes an all-constant column and teaches the
#: model the base rate.
MIN_DISTINCT_MIDS = 4

#: And enough observations to fill the longest feature window.
MIN_QUOTES = 120


def _finite(value: float | None) -> bool:
    return value is not None and value == value and abs(value) != float("inf")


def _book(bid_cc: int, ask_cc: int, bid_qc: int | None, ask_qc: int | None) -> OrderBook | None:
    """Rebuild a two-sided event book from one stored quote.

    Kalshi's NO side is the reflection of the YES side, so a YES ask of 41c is a
    NO bid of 59c. Reconstructing it that way rather than storing both keeps this
    book identical in shape to the one the live engine holds.
    """
    if bid_cc is None or ask_cc is None or not 0 < bid_cc < ask_cc < 10_000:
        return None
    return OrderBook.from_event_contract_levels(
        "x",
        yes_levels=[(bid_cc, bid_qc or 10_000)],
        no_levels=[(10_000 - ask_cc, ask_qc or 10_000)],
    )


async def fetch_settlements(tickers: list[str]) -> dict[str, float]:
    """How each market resolved, as 1.0 for YES and 0.0 for NO.

    Only markets that actually resolved appear. A market still open, or one that
    voided, has no outcome and must not be given one: SettlementLabel returns
    None for it and the row is dropped rather than labelled with a guess.
    """
    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "select ticker, result from settlements where ticker = any($1::text[])",
            tickers,
        )
    finally:
        await conn.close()
    return {
        r["ticker"]: (1.0 if r["result"] == "yes" else 0.0)
        for r in rows
        if r["result"] in ("yes", "no")
    }


async def fetch_close_times(tickers: list[str]) -> dict[str, int]:
    """Close times for the markets being trained on.

    ``time_to_expiry_ms`` is a required feature -- a contract five minutes from
    resolution behaves nothing like one five days out, and a model blind to the
    difference fits two regimes as one. It lives in the market definition, which
    the quotes table does not carry, so it is fetched.

    That it must be fetched at all is a gap worth naming: the engine already
    reads these definitions to feed the news matcher and then discards them,
    leaving the ``markets`` table empty. Persisting them would remove this
    lookup and let training run with no network at all.
    """
    from kat.kalshi.auth import KalshiSigner
    from kat.kalshi.event_contracts.rest import EventContractsREST
    from kat.kalshi.http import KalshiHTTPClient
    from kat.kalshi.ratelimit import RateLimiter

    settings = AppSettings()
    creds = settings.kalshi.active_credentials
    signer = KalshiSigner.from_credentials(creds, label="event") if creds.configured else None
    http = KalshiHTTPClient(
        base_url=settings.kalshi.rest_base_url,
        signer=signer,
        rate_limiter=RateLimiter(),
        is_perp=False,
    )
    rest = EventContractsREST(http)

    closes: dict[str, int] = {}
    missing = 0
    try:
        for ticker in tickers:
            try:
                market = await rest.get_market(ticker)
            except Exception:
                missing += 1
                continue
            if market.close_time is not None:
                closes[ticker] = int(market.close_time.timestamp() * 1000)
            else:
                missing += 1
    finally:
        await http.aclose()
    log.info("train.close_times", resolved=len(closes), unavailable=missing)
    return closes


async def load_histories(hours: int, limit_markets: int) -> dict[str, MarketHistory]:
    """Rebuild per-market history from the quotes table."""
    import asyncpg

    dsn = re.sub(r"^postgresql\+\w+", "postgresql", str(AppSettings().database.url))
    conn = await asyncpg.connect(dsn)
    try:
        since = datetime.now(UTC) - timedelta(hours=hours)
        candidates = await conn.fetch(
            """
            select ticker
            from quotes
            where market_type = 'event' and ts >= $1
              and bid_cc is not null and ask_cc is not null
            group by ticker
            having count(*) >= $2
               and count(distinct (bid_cc + ask_cc) / 2) >= $3
            order by count(*) desc
            limit $4
            """,
            since, MIN_QUOTES, MIN_DISTINCT_MIDS, limit_markets,
        )
        tickers = [r["ticker"] for r in candidates]
        log.info("train.markets_selected", markets=len(tickers), hours=hours)
        if not tickers:
            return {}

        rows = await conn.fetch(
            """
            select ticker, ts, bid_cc, ask_cc, bid_size_qc, ask_size_qc,
                   last_cc, volume_qc, open_interest_qc
            from quotes
            where market_type = 'event' and ts >= $1 and ticker = any($2::text[])
            order by ticker, ts
            """,
            since, tickers,
        )
    finally:
        await conn.close()

    books: dict[str, list[BookObservation]] = defaultdict(list)
    ticks: dict[str, list[TickerObservation]] = defaultdict(list)
    mids: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for row in rows:
        ts_ms = int(row["ts"].timestamp() * 1000)
        book = _book(row["bid_cc"], row["ask_cc"], row["bid_size_qc"], row["ask_size_qc"])
        if book is None:
            continue
        ticker = row["ticker"]
        books[ticker].append(BookObservation(ts_ms=ts_ms, book=book))
        if book.mid_cc is not None:
            mids[ticker].append((ts_ms, book.mid_cc))
        if row["volume_qc"] is not None or row["last_cc"] is not None:
            ticks[ticker].append(
                TickerObservation(
                    ts_ms=ts_ms,
                    volume_qc=row["volume_qc"],
                    open_interest_qc=row["open_interest_qc"],
                    last_trade_cc=row["last_cc"],
                )
            )

    usable = [t for t, obs in books.items() if len(obs) >= MIN_QUOTES]
    closes = await fetch_close_times(usable)
    settlements = await fetch_settlements(usable)

    histories: dict[str, MarketHistory] = {}
    for ticker in usable:
        close_ms = closes.get(ticker)
        if close_ms is None:
            # Dropped rather than defaulted. A guessed expiry puts a wrong number
            # into a required feature, and the model then learns a relationship
            # that is not there -- worse than a smaller dataset.
            continue
        histories[ticker] = MarketHistory(
            ticker,
            MarketType.EVENT,
            books=books[ticker],
            bars=bars_from_mid_path(mids[ticker], period_ms=60_000),
            tickers=ticks.get(ticker, []),
            close_ts_ms=close_ms,
            # None where the market has not resolved. SettlementLabel drops
            # those rows rather than labelling them, which is the difference
            # between "we do not know yet" and "it resolved NO".
            settlement_yes=settlements.get(ticker),
        )
    log.info(
        "train.histories_built",
        markets=len(histories),
        observations=sum(len(books[t]) for t in histories),
    )
    return histories


async def _record_run(
    database: Any,
    args: Any,
    dataset: Any,
    result: Any,
    *,
    registered: Any | None,
) -> None:
    """Leave a durable trace of this run, whatever it concluded.

    Written straight into a session rather than through :class:`AsyncWriter`.
    The writer is a long-lived background task that flushes on an interval; a
    process that exits seconds later would drop the row it was asked to persist,
    which is the one thing this must not do.

    The audit page renders ``detail`` as ``JSON.stringify(...).slice(0, 160)``
    with no expansion and no tooltip, and Postgres stores JSONB with its own key
    order -- so writing the important keys first does not keep them visible. The
    only thing that works is keeping the whole object under that budget, which
    is why this is four short keys and a truncated reason rather than everything
    the run knows. The full picture is in the Models page and in stdout.
    """
    from kat.db.models import AuditLog

    candidates = getattr(result, "candidates", []) or []
    if registered is not None:
        outcome = "deployed" if args.deploy else "registered"
        subject = registered.model_id
        # A retrain that changed what the platform will do is worth a colour.
        severity = "warning" if args.deploy else "info"
        detail: dict[str, Any] = {
            "out": outcome,
            "fam": result.best.name,
            "auc": round(result.best.test_metrics.get("auc", float("nan")), 3),
            "rows": dataset.n_rows,
        }
    else:
        outcome = "gates_refused"
        subject = "none"
        severity = "info"
        # The gate that stopped the best-scoring candidate, not all of them:
        # four families times three failures each does not fit in the budget,
        # and the first one is what tells you what to change.
        best = max(candidates, key=lambda c: c.score, default=None)
        reason = (best.failures[0] if best and best.failures else result.reason) or ""
        detail = {
            "out": outcome,
            "why": reason.split(" (")[0][:74],
            "rows": dataset.n_rows,
            "n": len(candidates),
        }

    async with database.session() as session:
        session.add(
            AuditLog(
                actor="retrain",
                action="model_retrained",
                subject=subject[:128],
                detail=detail,
                severity=severity,
                environment=AppSettings().environment.value,
            )
        )
    print(f"  recorded: audit entry model_retrained ({outcome})")


def _balance(y: Any) -> str:
    n = len(y)
    if not n:
        return "0"
    positives = int(sum(y))
    return f"{n} ({positives} pos, {positives / n:.1%})"


def _report(models: list[Any]) -> None:
    print()
    print(f"  {'family':<16}{'val AUC':>10}{'val Brier':>12}{'test AUC':>10}  note")
    for model in models:
        if model.failed:
            print(f"  {model.name:<16}{'--':>10}{'--':>12}{'--':>10}  FAILED: {model.failed[:40]}")
            continue
        val, test = model.validation_metrics, model.test_metrics
        auc, t_auc = val.get("auc"), test.get("auc")
        auc_s = f"{auc:.4f}" if _finite(auc) else "n/a"
        t_auc_s = f"{t_auc:.4f}" if _finite(t_auc) else "n/a"
        print(
            f"  {model.name:<16}{auc_s:>10}"
            f"{val.get('brier', float('nan')):>12.4f}{t_auc_s:>10}"
        )


def _report_walk_forward(dataset: Any, *, folds: int) -> None:
    """Score every family across rolling windows and say what the gap means.

    The single split reports one number for the train-to-test gap and cannot say
    where it came from. This runs the same families over rolling windows and
    reports the gap's *spread*, which is the part that distinguishes the two
    explanations:

    * a large gap in every window is the model memorising its training data --
      more data will not fix that, a simpler model might;
    * a gap that swings between windows is the market changing underneath it,
      and the single split simply landed on an unlucky period.

    The base-rate spread is printed beside it because that is the mechanism: if
    the fraction of positive labels moves between windows, the model is being
    asked a different question in each one.
    """
    from kat.models.trainers import TRAINERS, walk_forward

    print()
    print(f"  walk-forward, {folds} rolling windows")
    print(f"  {'family':<16}{'mean AUC':>10}{'worst':>8}{'mean gap':>10}{'gap spread':>12}{'scored':>8}")

    verdicts: list[tuple[str, float | None, float | None]] = []
    for name in TRAINERS:
        summary, fold_results = walk_forward(
            name,
            dataset.X,
            dataset.y,
            list(dataset.feature_names),
            timestamps=dataset.as_of_ms,
            folds=folds,
        )
        if not summary.get("folds"):
            print(f"  {name:<16}{summary.get('reason', 'no folds')}")
            continue

        def _fmt(value: float | None, places: int = 4) -> str:
            return "n/a" if value is None else f"{value:.{places}f}"

        print(
            f"  {name:<16}{_fmt(summary['mean_test_auc']):>10}"
            f"{_fmt(summary['worst_test_auc'], 3):>8}"
            f"{_fmt(summary['mean_gap']):>10}"
            f"{_fmt(summary['gap_spread']):>12}"
            f"{summary['scored']}/{summary['folds']:>3}"
        )
        verdicts.append((name, summary.get("mean_gap"), summary.get("gap_spread")))
        for fold in fold_results:
            if fold.failed:
                print(f"      fold {fold.index}: {fold.failed}")

    rates = [f for f in (verdicts or []) if f[2] is not None]
    if not rates:
        return
    widest = max(rates, key=lambda v: v[2] or 0.0)
    _, mean_gap, spread = widest
    print()
    if spread is not None and mean_gap is not None and spread > mean_gap:
        print(
            "  The gap varies more between windows than its own average, so the "
            "single split was measuring the market changing rather than the model "
            "memorising. More data across more regimes is the fix; a simpler model "
            "is not."
        )
    else:
        print(
            "  The gap is consistent across windows, so it is the model memorising "
            "rather than the market moving. More data will not fix that -- fewer "
            "features or firmer regularisation might."
        )

def _split(dataset: Any, args: Any) -> Any:
    """Split the dataset, purging label overlap across the boundaries.

    ``temporal_split`` orders by time and cuts, which is right for a fixed
    horizon and wrong for settlement. A settlement label is not knowable until
    the market closes, so two rows observed days apart about the same market
    carry the *same* outcome. Cut between them and the training set contains the
    test set's answers -- and a market watched for a week contributes dozens of
    such rows.

    ``chronological_split`` drops rows whose label reaches past the boundary and
    leaves a gap after it, which is what makes the test score mean anything. Its
    docstring asks for an explicit embargo when the horizon is variable, which is
    exactly the settlement case; a day is used, being longer than the gap between
    consecutive observations of one market and shorter than the history.
    """
    from kat.features.pipeline import chronological_split
    from kat.models.trainers import SplitResult

    if args.label != "settlement":
        return temporal_split(dataset.X, dataset.y, timestamps=dataset.as_of_ms)

    from kat.common.errors import DataQualityError

    try:
        splits = chronological_split(dataset, embargo_ms=24 * 60 * 60 * 1000)
    except DataQualityError as exc:
        # Not a crash: the data cannot be split honestly, and saying so is the
        # correct outcome. Measured on the first settlement dataset, purging
        # alone emptied the validation split even with a zero embargo -- almost
        # every observation's label resolved after the test window began,
        # because the settled markets all closed near the end of the history
        # while observations span the whole of it. A model fitted on that would
        # be scored on labels its training rows already contained.
        print(f"\n  cannot split this dataset honestly: {exc}", file=sys.stderr)
        print(
            "  Settlement labels need markets that resolve *throughout* the "
            "history, not all at the end of it. The settlement recorder runs "
            "continuously, so this becomes possible as more markets close.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(
        f"  purged   {splits.purged_rows} rows whose label crossed a split "
        f"boundary (embargo {splits.embargo_ms // 3_600_000}h)"
    )
    return SplitResult(
        X_train=splits.train.X, y_train=splits.train.y,
        X_val=splits.validation.X, y_val=splits.validation.y,
        X_test=splits.test.X, y_test=splits.test.y,
        train_end_index=int(splits.train.y.size),
        val_end_index=int(splits.train.y.size + splits.validation.y.size),
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Train and optionally deploy a model.")
    parser.add_argument("--hours", type=int, default=96, help="how much history to use")
    parser.add_argument("--markets", type=int, default=150, help="most-active markets to include")
    parser.add_argument("--horizon-min", type=int, default=30, help="forward window, minutes")
    parser.add_argument(
        "--threshold-cc",
        type=int,
        default=50,
        help="how far the mid must move to count. Zero would train on 'moved at "
             "all', which is not the question a strategy paying a spread asks.",
    )
    parser.add_argument("--step-min", type=int, default=1, help="minutes between decision times")
    parser.add_argument(
        "--label",
        choices=("forward_move", "settlement"),
        default="forward_move",
        help="what to predict. 'settlement' is the chance the contract resolves "
             "YES, which is the only target MispricingStrategy can read: it "
             "converts the probability into a fair value for the contract. "
             "'forward_move' predicts a mid move and must never be fed to that "
             "strategy, which is why the target now travels with the model.",
    )
    parser.add_argument("--deploy", action="store_true", help="deploy if it clears the gates")
    parser.add_argument(
        "--no-walk-forward",
        dest="walk_forward",
        action="store_false",
        help="skip the rolling-window diagnostic. It costs one extra fit per "
             "family per fold and is what explains a gate failure, so it is on "
             "by default.",
    )
    parser.add_argument("--folds", type=int, default=5, help="rolling windows")
    args = parser.parse_args()

    configure_logging("WARNING")

    histories = await load_histories(args.hours, args.markets)
    if not histories:
        print("  no markets with enough price variation to train on.", file=sys.stderr)
        return 1

    pipeline = FeaturePipeline(EVENT_FEATURE_SET)
    if args.label == "settlement":
        labeller: Any = SettlementLabel()
        target = "settlement_yes"
        described = "resolved YES"
    else:
        labeller = ForwardMoveLabel(
            horizon_ms=args.horizon_min * 60_000, threshold_cc=args.threshold_cc
        )
        target = f"forward_move_{args.horizon_min}m"
        described = f"move > {args.threshold_cc}cc within {args.horizon_min}m"
    dataset = pipeline.build_dataset(
        list(histories.values()),
        labeller=labeller,
        step_ms=args.step_min * 60_000,
        name=f"event_{target}",
    )

    print()
    print(f"  markets  {len(set(dataset.tickers))}")
    print(f"  dataset  rows={dataset.n_rows} features={dataset.n_features}")
    print(f"  label    {described}")
    print(f"  positive rate: {dataset.positive_rate:.4f}")
    if dataset.n_rows < 500:
        print("  too few rows to train on honestly; collect more data.", file=sys.stderr)
        return 1

    split = _split(dataset, args)
    print(
        f"  split    train={_balance(split.y_train)}  "
        f"val={_balance(split.y_val)}  test={_balance(split.y_test)}"
    )
    # A base rate that shifts hard between splits is the thing most likely to
    # sink calibration, and it is invisible in an averaged metric. Say it here.
    rates = [float(sum(y)) / len(y) for y in (split.y_train, split.y_val, split.y_test) if len(y)]
    if rates and max(rates) - min(rates) > 0.15:
        print(
            "  WARNING: the base rate moves sharply across the splits, so the test "
            "window is a different regime from the training one. Expect calibration "
            "to fail its gate; that is the data speaking, not the model."
        )

    models = train_all(split, list(dataset.feature_names))
    _report(models)

    if args.walk_forward:
        _report_walk_forward(dataset, folds=args.folds)

    trained = [m for m in models if m.failed is None]
    if not trained:
        print("\n  every family failed to train.", file=sys.stderr)
        return 1

    # The platform's own gates, not a hand-rolled pick.
    #
    # select_best is entitled to return nothing, and that refusal is the feature:
    # deploying the least bad of several failing models is how a platform ends up
    # trading on noise. When it refuses, the answer is more data or a better
    # target -- never a lower gate.
    result = select_best(trained)
    print("\n  selection gates")
    for candidate in result.candidates:
        mark = "PASS" if candidate.passed else "fail"
        why = "" if candidate.passed else "  <- " + "; ".join(candidate.failures)
        print(f"    {candidate.model.name:<16}{mark}  score={candidate.score:.4f}{why}")

    # The database is opened before the outcome is known, on purpose.
    #
    # This used to return early when every family failed its gates, before the
    # connection was made, so the most likely outcome of an unattended retrain
    # left no record anywhere. The Models page would show "No models registered
    # yet" -- which is exactly what it shows when nothing ever ran. A scheduled
    # job whose commonest result is indistinguishable from not having run is not
    # reporting anything.
    database = get_database()
    await database.connect()
    try:
        if result.best is None:
            print("\n  NOTHING DEPLOYED: no family cleared the gates.", file=sys.stderr)
            print(f"  {result.reason}", file=sys.stderr)
            await _record_run(database, args, dataset, result, registered=None)
            return 2

        winner = result.best
        print(f"\n  winner: {winner.name}")
        print(f"    validation: {winner.validation_metrics}")
        print(f"    test:       {winner.test_metrics}")

        registry = ModelRegistry(database)
        registered = await registry.register(
            winner,
            market_type=MarketType.EVENT,
            # The same target the dataset was labelled with. Hard-coding
            # the forward-move name here would register a settlement model
            # under a target it does not predict, and the whole point of
            # the field is that consumers can tell what the number means.
            target=target,
            dataset_id=dataset.dataset_id,
            feature_set_id=EVENT_FEATURE_SET.feature_set_id,
        )
        print(f"\n  registered: {registered.model_id} v{registered.version}")

        if args.deploy:
            await registry.deploy(registered.model_id, registered.version)
            print(f"  DEPLOYED:   {registered.model_id} v{registered.version}")
            print("  restart the engine to pick it up.")
        else:
            print("  not deployed. Re-run with --deploy when you want it live.")
        await _record_run(database, args, dataset, result, registered=registered)
    finally:
        await database.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
