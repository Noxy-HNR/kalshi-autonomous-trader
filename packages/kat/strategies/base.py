"""Strategy interface and lifecycle.

A strategy's only job is to form an opinion. It receives market state and
features, and it returns zero or more :class:`~kat.risk.types.OrderIntent`
objects. It does not size positions against the account, it does not check
risk, and it has no route to the exchange:

    Strategy -> Portfolio/Allocation -> Risk Engine -> Execution Engine -> Kalshi

The suggested size on an intent is a *preference*, expressed in the strategy's
own terms. Portfolio allocation scales it to available capital and the risk
engine has the final word on whether it is permitted at all. A strategy that
returns an enormous size is not dangerous; it is simply going to be resized.

This is enforced structurally rather than by convention: nothing passed to
:meth:`Strategy.generate` can reach the network, and the execution engine only
accepts a ``RiskVerdict``, which a strategy cannot produce.

Lifecycle
---------
Every strategy carries an explicit :class:`~kat.common.enums.StrategyState`.
Promotion toward ``LIVE`` moves one rung at a time and only when the gate for
that rung passes; demotion to ``PAUSED`` or ``RETIRED`` is always allowed::

    DISCOVERY -> DEVELOPMENT -> BACKTEST -> OUT_OF_SAMPLE -> WALK_FORWARD
      -> ROBUSTNESS -> PAPER -> PROBATION -> LIVE -> MONITORING -> REVIEW
      -> RETIRED

In Mode D (the default) no strategy reaches ``LIVE`` without clearing every
gate. See :mod:`kat.discovery.lifecycle`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import MarketType, StrategyState
from kat.marketdata.state import MarketSnapshot
from kat.risk.types import OrderIntent

__all__ = [
    "PROMOTION_LADDER",
    "Strategy",
    "StrategyContext",
    "StrategyMetadata",
    "StrategyParam",
    "next_promotion_state",
]

#: The order in which a strategy may be promoted. Index position is the rung.
PROMOTION_LADDER: tuple[StrategyState, ...] = (
    StrategyState.DISCOVERY,
    StrategyState.DEVELOPMENT,
    StrategyState.BACKTEST,
    StrategyState.OUT_OF_SAMPLE,
    StrategyState.WALK_FORWARD,
    StrategyState.ROBUSTNESS,
    StrategyState.PAPER,
    StrategyState.PROBATION,
    StrategyState.LIVE,
)


def next_promotion_state(current: StrategyState) -> StrategyState | None:
    """The single next rung, or ``None`` when there is nowhere further to go.

    Returns ``None`` for states outside the ladder (``PAUSED``, ``RETIRED``,
    ``MONITORING``, ``REVIEW``) so that a strategy cannot be promoted out of
    them by accident; re-entry is an explicit transition.
    """
    try:
        index = PROMOTION_LADDER.index(current)
    except ValueError:
        return None
    if index + 1 >= len(PROMOTION_LADDER):
        return None
    return PROMOTION_LADDER[index + 1]


@dataclass(frozen=True, slots=True)
class StrategyParam:
    """One tunable parameter and the range the optimiser may explore.

    Bounds are mandatory. An unbounded search space is how parameter
    optimisation discovers a strategy that holds 400 contracts of an illiquid
    market, and stating the range forces the author to think about what is
    reasonable before the optimiser does it for them.
    """

    name: str
    default: float | int | bool | str
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""

    def clamp(self, value: Any) -> Any:
        if self.choices is not None:
            return value if value in self.choices else self.default
        if isinstance(value, int | float):
            if self.low is not None:
                value = max(self.low, value)
            if self.high is not None:
                value = min(self.high, value)
        return value


@dataclass(slots=True)
class StrategyMetadata:
    """Identity and provenance of a strategy instance."""

    strategy_id: str
    version: str
    name: str
    market_type: MarketType
    family: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    param_space: tuple[StrategyParam, ...] = ()
    discovered_by: str = "human"
    git_commit: str | None = None
    # Minimum data a decision needs. The engine will not call a strategy whose
    # requirements it cannot satisfy, rather than letting it improvise.
    requires_book: bool = True
    requires_features: tuple[str, ...] = ()
    max_data_age_ms: int = 5_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "name": self.name,
            "market_type": self.market_type.value,
            "family": self.family,
            "description": self.description,
            "params": dict(self.params),
            "param_space": [
                {
                    "name": p.name, "default": p.default, "low": p.low, "high": p.high,
                    "step": p.step, "choices": list(p.choices) if p.choices else None,
                    "description": p.description,
                }
                for p in self.param_space
            ],
            "discovered_by": self.discovered_by,
            "git_commit": self.git_commit,
            "requires_features": list(self.requires_features),
        }


@dataclass(slots=True)
class StrategyContext:
    """Everything a strategy is allowed to see when forming an opinion.

    Note what is absent: no REST client, no execution engine, no account
    balance, no way to place an order. A strategy sees the market and its own
    position, and that is all it needs to have a view.

    ``now_ms`` comes from the platform clock, which is simulated during a
    backtest. A strategy that reads the wall clock instead would see the future.
    """

    now_ms: int
    market: MarketSnapshot
    # Feature values already filtered to those available at ``now_ms``.
    features: dict[str, float] = field(default_factory=dict)
    # Model probability for this market, when a model is deployed for it.
    model_probability: float | None = None
    #: What ``model_probability`` is a probability *of* -- the deployed model's
    #: target. A strategy that reads the number without checking this is not
    #: reading a probability, it is reading a float.
    model_target: str | None = None
    #: The unconditional rate of that event. A rare-event probability is not
    #: centred on a half, so this is the neutral point a directional reading has
    #: to be taken against.
    model_base_rate: float | None = None
    model_id: str | None = None
    model_version: str | None = None
    # Current signed position in this market, in centi-contracts.
    position_qc: int = 0
    #: How long the current position has been held, or None when flat or
    #: unknown. A forecast with a horizon is only actionable for that horizon.
    position_age_ms: int | None = None
    #: Quantity this strategy already has resting in this market. A quoting
    #: strategy that cannot see its own working orders re-quotes every cycle and
    #: stacks hundreds of them into one position instead of cycling.
    resting_qty_qc: int = 0
    # Detected regime, when regime detection is running.
    regime: str | None = None
    # Relevant news/research signals, already time-filtered.
    research: dict[str, Any] = field(default_factory=dict)
    # Fee schedule in force, so expected value can be computed net of costs.
    fee_multiplier: float | None = None
    #: Whether the feed itself is delivering, decided by the component that owns
    #: the connection rather than re-derived from this snapshot's age.
    #:
    #: A strategy cannot tell the two kinds of silence apart from a timestamp
    #: alone. On a delta feed a market that has sent nothing is usually saying
    #: the book is unchanged; the same age with the connection dropped means the
    #: opposite. Only :class:`~kat.marketdata.state.MarketState` knows which,
    #: because only it sees every market at once.
    #:
    #: ``None`` means "not supplied" -- backtests construct contexts directly --
    #: and the age check then stands alone, which is the older, stricter rule.
    feed_live: bool | None = None

    def feature(self, name: str, default: float = 0.0) -> float:
        return self.features.get(name, default)


class Strategy(ABC):
    """Base class for every strategy in both domains.

    Subclasses implement :meth:`generate` and describe themselves through
    :meth:`metadata`. Everything else -- lifecycle, allocation, risk, execution,
    measurement -- is done for them by the platform.
    """

    def __init__(self, metadata: StrategyMetadata, state: StrategyState = StrategyState.DEVELOPMENT) -> None:
        self._metadata = metadata
        self.state = state
        # Capital the portfolio layer has allocated, in micro-dollars. A
        # strategy may read it to scale its own conviction, but it is not a
        # limit the strategy is trusted to respect -- the risk engine enforces
        # the actual bound.
        self.allocated_capital_micros: int = 0

    # -- identity ------------------------------------------------------------

    @property
    def metadata(self) -> StrategyMetadata:
        return self._metadata

    @property
    def strategy_id(self) -> str:
        return self._metadata.strategy_id

    @property
    def version(self) -> str:
        return self._metadata.version

    @property
    def market_type(self) -> MarketType:
        return self._metadata.market_type

    @property
    def params(self) -> dict[str, Any]:
        return self._metadata.params

    def param(self, name: str, default: Any = None) -> Any:
        return self._metadata.params.get(name, default)

    # -- the one thing a strategy must implement -----------------------------

    @abstractmethod
    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        """Return the intents this strategy would like to act on, if any.

        Returning an empty list is the normal, expected outcome. The platform
        optimises for risk-adjusted return, not activity, and a strategy that
        always finds something to do is a strategy that has stopped being
        selective.

        Implementations must be **pure with respect to time**: they may read
        ``ctx.now_ms`` but never the wall clock, and they may read only the
        features handed to them, which the caller has already filtered to what
        was knowable at ``now_ms``. This is what makes the same code valid in a
        backtest and in live trading.
        """

    # -- optional hooks ------------------------------------------------------

    def on_fill(self, ticker: str, qty_qc: int, price_cc: int) -> None:  # noqa: B027
        """Called after one of this strategy's orders fills.

        Deliberately concrete and empty rather than abstract: most strategies do
        not care, and forcing every one of them to write ``pass`` would be noise.
        """

    def on_position_closed(self, ticker: str, realized_pnl_micros: int) -> None:  # noqa: B027
        """Called when a position opened by this strategy is fully closed.

        Optional hook; see :meth:`on_fill` for why it is not abstract.
        """

    def warmup_ms(self) -> int:
        """How much history this strategy needs before its signals are valid.

        The simulator refuses to score trades taken inside the warm-up window,
        so a strategy that needs an hour of data does not get credit for a
        decision made in its first minute.
        """
        return 0

    def health(self) -> dict[str, Any]:
        """Optional self-reported diagnostics for the dashboard."""
        return {}

    # -- presentation --------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.strategy_id!r}, v={self.version!r}, "
            f"state={self.state.value})"
        )
