"""Trade a perpetual against the index it is supposed to track.

A perpetual future has no delivery date, so nothing forces its price to the
underlying index -- except funding, which is charged in proportion to the gap
and in the direction that closes it. That makes the basis the one perpetual
signal with an actual mechanism behind it rather than a statistical regularity:
when the perpetual trades above the index, longs pay, and paying to hold a rich
position is what eventually makes it less rich.

The two edges point the same way, which is the appeal: a short in a
richly-priced perpetual is paid funding *while* waiting for the convergence it
is betting on. Both are folded into one expected value here rather than being
counted as separate reasons to like the trade, because counting them separately
is how a marginal trade gets talked into being a good one.

The index is an input, and inputs break
---------------------------------------
The basis is the difference between two numbers from two different places, so a
fault in either one shows up as a large, entirely fictitious opportunity -- and
the larger the fault, the more attractive the trade looks. The gates are built
around that, not around the trade being profitable:

* ``max_basis_cc`` is an **upper** bound. A gap wider than this is treated as
  evidence that the index is stale or wrong and produces no order at all. This
  is the single most important line in the module: without it, a frozen index
  feed reads as an ever-growing edge.
* ``max_index_age_ms`` rejects an index whose published age says it has stopped
  updating, when the feature engine supplies that age.
* The relative gate (``min_basis_bps``) sits alongside the absolute one so the
  same parameters behave sensibly for a contract at five cents and one at ninety.

Feature contract
----------------
``index_price_cc`` is required, in centicents, on the same scale as the
perpetual's own price. ``index_price_age_ms`` is optional; when the feature
engine publishes it, it is enforced, and when it does not, the point-in-time
guarantee of the feature bus is the only freshness assurance available and the
intent records that fact. As a fallback the index may instead be carried on the
market snapshot's metadata under the same key, which is how a venue-published
index that never becomes a feature can still be used.
"""

from __future__ import annotations

from kat.common.enums import MarketType
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import TradingStrategy, clamp_price_cc
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    LIQUIDITY_PARAM_SPACE,
    conservative_funding_rate,
    net_ev_floor_micros,
    propose_perp_trade,
    screen_freshness,
    screen_perp_market,
)

__all__ = ["F_INDEX_AGE_MS", "F_INDEX_PRICE_CC", "PerpBasisStrategy"]

#: Reference/index price for the perpetual, in centicents. Declared here rather
#: than in :mod:`kat.strategies.common` because this is the only consumer, and a
#: feature-name constant with one user belongs next to that user.
F_INDEX_PRICE_CC = "index_price_cc"
#: Age of that index observation in milliseconds. Optional; enforced when present.
F_INDEX_AGE_MS = "index_price_age_ms"

_SKIP_NO_INDEX = "index_price_unavailable"
_SKIP_INDEX_STALE = "index_price_stale"
_SKIP_NO_PRICE = "no_reference_price"
_SKIP_BASIS_TOO_SMALL = "basis_below_threshold"
_SKIP_BASIS_IMPLAUSIBLE = "basis_too_large_to_believe"
_SKIP_NO_MID = "no_midpoint"

_MS_PER_HOUR = 3_600_000.0


class PerpBasisStrategy(TradingStrategy):
    """Fade the gap between the perpetual mark and its index."""

    FAMILY = "basis"
    NAME = "Perpetual Basis Convergence"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.PERP
    DESCRIPTION = """
    Trades the perpetual against its published index when the gap is large
    enough to clear costs but small enough to be believable, taking the side
    that funding also pays, and forecasting only a fraction of convergence.
    """
    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "min_basis_cc", 100, low=10, high=2_000, step=10,
            description="absolute gap required between mark and index, in centicents",
        ),
        StrategyParam(
            "min_basis_bps", 100, low=0, high=2_000, step=25,
            description="gap required as a fraction of the index, in basis points",
        ),
        StrategyParam(
            "max_basis_cc", 1_500, low=100, high=5_000, step=100,
            description="gap above which the index is assumed broken and no order is sent",
        ),
        StrategyParam(
            "max_index_age_ms", 60_000, low=1_000, high=900_000, step=1_000,
            description="index observations older than this are refused, when age is published",
        ),
        StrategyParam(
            "convergence_fraction", 0.4, low=0.05, high=1.0, step=0.05,
            description="fraction of the basis expected to close over the holding period",
        ),
        StrategyParam(
            "max_convergence_cc", 400, low=25, high=3_000, step=25,
            description="hard cap on how far the forecast may sit from the midpoint",
        ),
        StrategyParam(
            "hold_hours", 4.0, low=0.5, high=72.0, step=0.5,
            description="intended holding period, over which funding is integrated",
        ),
        StrategyParam(
            "rate_persistence", 0.5, low=0.05, high=1.0, step=0.05,
            description="fraction of the current funding rate assumed to persist, when it is income",
        ),
    )
    REQUIRES_FEATURES = (F_INDEX_PRICE_CC,)
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def warmup_ms(self) -> int:
        """Point-in-time comparison of two prices; no history of its own."""
        return 0

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        index_cc, age_known = self._index_price_cc(ctx)
        if index_cc is None:
            return self.skip(_SKIP_NO_INDEX)
        if age_known is False and F_INDEX_AGE_MS in ctx.features:
            return self.skip(_SKIP_INDEX_STALE)

        mark_cc = ctx.market.reference_price_cc()
        if mark_cc is None or mark_cc <= 0:
            return self.skip(_SKIP_NO_PRICE)

        basis_cc = mark_cc - index_cc
        if abs(basis_cc) > self.int_param("max_basis_cc"):
            return self.skip(_SKIP_BASIS_IMPLAUSIBLE)

        relative_floor_cc = index_cc * self.int_param("min_basis_bps") / 10_000.0
        if abs(basis_cc) < max(float(self.int_param("min_basis_cc")), relative_floor_cc):
            return self.skip(_SKIP_BASIS_TOO_SMALL)

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

        # A perpetual trading above its index is expected to fall toward it.
        is_long = basis_cc < 0
        displacement = -self.float_param("convergence_fraction") * basis_cc
        cap_cc = float(self.int_param("max_convergence_cc"))
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
            confidence=self._confidence(basis_cc),
            costing=proposal.costing,
            reason="basis_convergence",
            extra={
                "index_price_cc": index_cc,
                "index_age_enforced": F_INDEX_AGE_MS in ctx.features,
                "mark_price_cc": mark_cc,
                "basis_cc": basis_cc,
                "forecast_mid_cc": forecast_mid_cc,
                "hold_hours": hold_hours,
                "max_hold_ms": int(hold_hours * _MS_PER_HOUR),
                "assumed_rate_bps_per_hour": round(assumed_rate, 6),
                "market_mid_cc": mid_cc,
            },
        )
        return self.emit([intent])

    # -- internals -----------------------------------------------------------

    def _index_price_cc(self, ctx: StrategyContext) -> tuple[int | None, bool]:
        """The index price and whether its published age is acceptable.

        The second element is ``True`` when no age was published (nothing to
        reject on) or the published age is inside the limit, and ``False`` only
        when an age was published and is too old.
        """
        raw = ctx.features.get(F_INDEX_PRICE_CC)
        if raw is None:
            meta = ctx.market.metadata.get(F_INDEX_PRICE_CC)
            raw = float(meta) if isinstance(meta, int | float) else None
        if raw is None or raw <= 0:
            return None, True

        age = ctx.features.get(F_INDEX_AGE_MS)
        fresh = age is None or age <= self.int_param("max_index_age_ms")
        return clamp_price_cc(int(raw)), fresh

    def _confidence(self, basis_cc: int) -> float:
        """Half conviction at the entry threshold, full at twice it."""
        floor_cc = max(1, self.int_param("min_basis_cc"))
        return min(1.0, abs(basis_cc) / (2.0 * floor_cc))
