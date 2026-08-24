"""Trend following on the perpetual mark, with the carry it will pay charged up front.

Why the trend is measured in time, not in ticks
-----------------------------------------------
Perpetual marks update continuously and irregularly. A moving average over "the
last N updates" measures a different amount of history depending on how busy the
market is, so a parameter tuned during a quiet hour means something else during
a busy one, and the strategy silently becomes faster exactly when the market
becomes noisier. :class:`~kat.strategies.common.PriceTracker` decays its weights
with elapsed milliseconds instead, driven by ``ctx.now_ms``, so a time constant
means the same thing at every level of activity and a backtest reproduces
exactly.

Why the signal is standardised
------------------------------
A gap between a fast and a slow average is a number of centicents, and a number
of centicents is meaningless without knowing how far this market normally moves.
The gap is therefore divided by the realised volatility the tracker estimates
from the same series. That makes the entry threshold a z-score, comparable
across markets and across regimes, and it makes the strategy stand down when
volatility rises without the trend strengthening -- which is the situation in
which trend following loses most of its money.

Why funding is a cost here
--------------------------
A trend position on a perpetual pays or receives funding for as long as it is
held, and a trend that is expected to pay for itself over several hours must
clear the funding it will pay over those hours. It is folded into the expected
value with :func:`~kat.strategies.proposal.conservative_funding_rate`, which
discounts income but never a cost.

State
-----
This strategy is stateful: the tracker accumulates across calls. That is what
lets it work from a stream rather than requiring the feature engine to publish
per-strategy EMAs, but it means a fresh instance (or :meth:`reset`) is required
between backtest runs, or the second run begins with the first one's memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kat.common.enums import MarketType
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import PriceTracker, TradingStrategy, clamp_price_cc
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    LIQUIDITY_PARAM_SPACE,
    conservative_funding_rate,
    net_ev_floor_micros,
    propose_perp_trade,
    screen_freshness,
    screen_perp_market,
)

__all__ = ["PerpTrendStrategy"]

_SKIP_NO_PRICE = "no_reference_price"
_SKIP_WARMING_UP = "tracker_still_warming_up"
_SKIP_NO_VOL = "realized_vol_unavailable"
_SKIP_SIGNAL_TOO_SMALL = "trend_below_threshold"
_SKIP_NO_MID = "no_midpoint"

_MS_PER_HOUR = 3_600_000.0
#: An exponential average is a meaningful estimate after roughly three time
#: constants; before that it is mostly its own initial value.
_WARMUP_TAU_MULTIPLE = 3


class PerpTrendStrategy(TradingStrategy):
    """Follow a volatility-standardised trend in the perpetual mark."""

    FAMILY = "trend"
    NAME = "Perpetual Mark Trend"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.PERP
    DESCRIPTION = """
    Follows the gap between a fast and a slow time-decayed average of the mark,
    standardised by the realised volatility of the same series. Claims a
    fraction of the gap as the forecast, charges the funding the position will
    pay over its intended life, and refuses while the tracker is warming up.
    """
    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "fast_tau_ms", 300_000, low=30_000, high=3_600_000, step=30_000,
            description="time constant of the fast average, in milliseconds",
        ),
        StrategyParam(
            "slow_tau_ms", 1_800_000, low=120_000, high=21_600_000, step=60_000,
            description="time constant of the slow average, in milliseconds",
        ),
        StrategyParam(
            "vol_tau_ms", 1_800_000, low=300_000, high=21_600_000, step=300_000,
            description="time constant of the realised-volatility estimate, in milliseconds",
        ),
        StrategyParam(
            "entry_z", 1.0, low=0.25, high=5.0, step=0.05,
            description="fast/slow gap required, in hourly standard deviations",
        ),
        StrategyParam(
            "trend_capture", 0.5, low=0.05, high=2.0, step=0.05,
            description="fraction of the fast/slow gap taken as the forecast displacement",
        ),
        StrategyParam(
            "max_forecast_cc", 300, low=25, high=3_000, step=25,
            description="hard cap on how far the forecast may sit from the midpoint",
        ),
        StrategyParam(
            "hold_hours", 6.0, low=0.5, high=72.0, step=0.5,
            description="intended holding period, used to charge funding against the trade",
        ),
        StrategyParam(
            "rate_persistence", 0.5, low=0.05, high=1.0, step=0.05,
            description="fraction of the current funding rate assumed to persist, when it is income",
        ),
    )
    REQUIRES_FEATURES = ()
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def __init__(self, params: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self._last_ticker: str | None = None
        self._tracker = self._new_tracker()

    def _new_tracker(self) -> PriceTracker:
        return PriceTracker(
            fast_tau_ms=self.int_param("fast_tau_ms"),
            slow_tau_ms=self.int_param("slow_tau_ms"),
            vol_tau_ms=self.int_param("vol_tau_ms"),
        )

    def reset(self) -> None:
        """Drop all accumulated price history.

        Required between backtest runs of the same instance; without it the
        second run starts with averages fitted to the first one's data.
        """
        self._tracker = self._new_tracker()
        self._last_ticker = None

    def warmup_ms(self) -> int:
        return _WARMUP_TAU_MULTIPLE * self.int_param("slow_tau_ms")

    def health(self) -> dict[str, Any]:
        base = super().health()
        base["last_ticker"] = self._last_ticker
        base["observed_ms"] = self._tracker.observed_ms(self._last_ticker) if self._last_ticker else 0
        base["warmup_ms"] = self.warmup_ms()
        return base

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        price_cc = ctx.market.reference_price_cc()
        if price_cc is None or price_cc <= 0:
            return self.skip(_SKIP_NO_PRICE)

        ticker = ctx.market.ticker
        self._last_ticker = ticker
        # Fold the observation in before any tradeability screen, so that a
        # market we decline to trade today still has usable history tomorrow.
        series = self._tracker.update(ticker, price_cc, ctx.now_ms)
        if series is None:
            return self.skip(_SKIP_NO_PRICE)

        if self._tracker.observed_ms(ticker) < self.warmup_ms():
            return self.skip(_SKIP_WARMING_UP)

        vol_bps_per_hour = self._tracker.vol_bps_per_hour(ticker)
        if vol_bps_per_hour is None or vol_bps_per_hour <= 0.0:
            return self.skip(_SKIP_NO_VOL)

        gap_cc = series.ema_fast - series.ema_slow
        # Hourly standard deviation of this market, expressed in centicents.
        sigma_cc = price_cc * (vol_bps_per_hour / 10_000.0)
        z_score = gap_cc / sigma_cc if sigma_cc > 0.0 else 0.0
        if abs(z_score) < self.float_param("entry_z"):
            return self.skip(_SKIP_SIGNAL_TOO_SMALL)

        book, reason = screen_perp_market(
            ctx,
            max_spread_cc=self.int_param("max_spread_cc"),
            min_depth_qc=self.int_param("min_depth_qc"),
        )
        if book is None or reason is not None:
            return self.skip(reason or "no_book")

        mid_cc = book.mid_cc
        if mid_cc is None:
            return self.skip(_SKIP_NO_MID)

        is_long = gap_cc > 0.0
        displacement = self.float_param("trend_capture") * gap_cc
        cap_cc = float(self.int_param("max_forecast_cc"))
        displacement = max(-cap_cc, min(cap_cc, displacement))
        forecast_mid_cc = clamp_price_cc(int(mid_cc + displacement))

        hold_hours = self.float_param("hold_hours")
        assumed_rate = conservative_funding_rate(
            ctx.market.funding_rate,
            is_long=is_long,
            persistence=self.float_param("rate_persistence"),
        )

        proposal, reason = propose_perp_trade(
            self,
            ctx,
            book=book,
            is_long=is_long,
            forecast_mid_cc=forecast_mid_cc,
            max_position_qc=self.int_param("max_position_qc"),
            min_net_ev_micros=net_ev_floor_micros(self.float_param("min_net_ev_usd")),
            funding_rate_bps_per_hour=assumed_rate,
            hold_hours=hold_hours,
        )
        if proposal is None:
            return self.skip(reason or "no_proposal")

        intent = self.perp_intent(
            ctx,
            is_buy=proposal.is_long,
            qty_qc=proposal.qty_qc,
            limit_price_cc=proposal.limit_price_cc,
            fair_value_cc=proposal.fair_value_cc,
            confidence=self._confidence(z_score),
            costing=proposal.costing,
            reason="mark_trend",
            extra={
                "ema_fast_cc": round(series.ema_fast, 2),
                "ema_slow_cc": round(series.ema_slow, 2),
                "gap_cc": round(gap_cc, 2),
                "z_score": round(z_score, 3),
                "vol_bps_per_hour": round(vol_bps_per_hour, 2),
                "forecast_mid_cc": forecast_mid_cc,
                "hold_hours": hold_hours,
                "max_hold_ms": int(hold_hours * _MS_PER_HOUR),
                "assumed_rate_bps_per_hour": round(assumed_rate, 6),
                "market_mid_cc": mid_cc,
            },
        )
        return self.emit([intent])

    # -- internals -----------------------------------------------------------

    def _confidence(self, z_score: float) -> float:
        entry_z = max(0.01, self.float_param("entry_z"))
        return min(1.0, abs(z_score) / (2.0 * entry_z))
