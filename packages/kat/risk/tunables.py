"""Risk parameters an operator may adjust while the platform runs.

Two kinds of number live in the risk configuration and this module exists to
keep them apart.

**Tunable.** How much of the account one position may use, how wide a spread the
strategies will cross, how much may be lost in a day. These are preferences.
Reasonable operators pick different values, the right value changes with account
size and market conditions, and getting one wrong costs money slowly and
visibly.

**Invariant.** The 1.00x perpetual leverage ceiling. It is not in this registry
and there is no code path that could put it here: the enforcement in
:mod:`kat.risk.invariants` is a comparison between two integers with no limit
parameter to read. It appears in :func:`locked_parameters` purely so the
dashboard can show it next to the sliders and say, truthfully, that there is no
slider for it.

Bounds are the point
--------------------
Every tunable carries a hard ``minimum`` and ``maximum`` taken from the clamps
already enforced in :mod:`kat.config.settings`. A slider cannot be dragged
outside them and the API refuses a value outside them, so widening a limit stays
a choice within a range someone thought about rather than an unbounded one.
Where the two disagree the settings clamp wins, because it runs on every load
including the ones this module never sees.

Changing one is an audited event
--------------------------------
Each change is written to the audit log with the old value, the new value and
who asked, and persisted so a restart does not silently revert it. A risk limit
that quietly returns to its default overnight is worse than one that was never
changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from kat.common.enums import MarketType
from kat.config.settings import AppSettings
from kat.risk.invariants import MAX_PERP_LEVERAGE

__all__ = [
    "TUNABLES",
    "Tunable",
    "TunableError",
    "apply_override",
    "current_values",
    "locked_parameters",
    "tunable_by_key",
]


class TunableError(ValueError):
    """A requested change is not one this platform will make."""


Kind = Literal["percent", "cents", "contracts", "count", "usd", "micros"]


@dataclass(frozen=True, slots=True)
class Tunable:
    """One adjustable risk parameter."""

    key: str
    domain: str  # "event" | "perp"
    field: str
    label: str
    #: What moving it actually does, in the operator's terms rather than the
    #: field name's. This is the text the dashboard shows under the slider.
    description: str
    kind: Kind
    minimum: Decimal
    maximum: Decimal
    step: Decimal
    #: True when a larger number means more risk. The dashboard colours the
    #: track by this, so an operator can see which way "riskier" is without
    #: reading each description.
    higher_is_riskier: bool = True

    def settings_block(self, settings: AppSettings) -> Any:
        return settings.event_risk if self.domain == "event" else settings.perp_risk

    def read(self, settings: AppSettings) -> Decimal:
        return Decimal(str(getattr(self.settings_block(settings), self.field)))

    def coerce(self, value: Any) -> Decimal | int:
        """Validate and convert an incoming value, or explain the refusal."""
        try:
            number = Decimal(str(value))
        except Exception as exc:  # the message is the product here
            raise TunableError(f"{self.key}: {value!r} is not a number") from exc
        if not number.is_finite():
            raise TunableError(f"{self.key}: {value!r} is not a finite number")
        if number < self.minimum or number > self.maximum:
            raise TunableError(
                f"{self.key}: {number} is outside the permitted range "
                f"[{self.minimum}, {self.maximum}]"
            )
        # Integer-valued fields must not acquire a fractional part on the way in.
        if self.kind in ("cents", "contracts", "count", "micros"):
            if number != number.to_integral_value():
                raise TunableError(f"{self.key}: {number} must be a whole number")
            return int(number)
        return number

    def to_dict(self, settings: AppSettings, defaults: AppSettings | None = None) -> dict[str, Any]:
        # The shipped default travels with the value because it is the only
        # meaningful reference for "is this setting aggressive?". Position
        # within the min/max range is not: several of these ranges are wide on
        # purpose, so a perfectly ordinary default can sit near a bound and read
        # as extreme.
        reference = defaults if defaults is not None else AppSettings()
        return {
            "default": str(self.read(reference)),
            "key": self.key,
            "domain": self.domain,
            "label": self.label,
            "description": self.description,
            "kind": self.kind,
            "value": str(self.read(settings)),
            "minimum": str(self.minimum),
            "maximum": str(self.maximum),
            "step": str(self.step),
            "higher_is_riskier": self.higher_is_riskier,
            "locked": False,
        }


def _d(value: str) -> Decimal:
    return Decimal(value)


#: The adjustable set. Bounds mirror the clamps in kat.config.settings; where
#: they differ the clamp wins, because it runs on every load.
TUNABLES: tuple[Tunable, ...] = (
    # -- what makes a market worth trading at all ---------------------------
    Tunable(
        key="event.max_spread_cc",
        domain="event",
        field="max_spread_cc",
        label="Widest spread to trade",
        description=(
            "How far apart the bid and ask may be before a market is treated as "
            "untradeable. The spread is paid on entry and again on exit, so this "
            "is the single largest cost a strategy faces. 300 = 3 cents."
        ),
        kind="cents",
        minimum=_d("1"),
        maximum=_d("5000"),
        step=_d("25"),
    ),
    Tunable(
        key="event.min_book_depth_qc",
        domain="event",
        field="min_book_depth_qc",
        label="Minimum book depth",
        description=(
            "Contracts that must be resting before a market counts as liquid "
            "enough to enter. 1000 = 10 contracts. Lowering it admits thinner "
            "markets where an exit may not be available at any sensible price."
        ),
        kind="contracts",
        minimum=_d("1"),
        maximum=_d("100000"),
        step=_d("100"),
        higher_is_riskier=False,
    ),
    Tunable(
        key="event.min_edge_cc",
        domain="event",
        field="min_edge_cc",
        label="Minimum edge required",
        description=(
            "How far from fair value a price must be before it is worth acting "
            "on. 200 = 2 cents. Lowering it produces more trades, each with less "
            "margin for the fair-value estimate to be wrong."
        ),
        kind="cents",
        minimum=_d("1"),
        maximum=_d("5000"),
        step=_d("25"),
        higher_is_riskier=False,
    ),
    Tunable(
        key="event.max_slippage_cc",
        domain="event",
        field="max_slippage_cc",
        label="Slippage tolerance",
        description=(
            "How far the price may move against the order between the decision "
            "and the fill before it is abandoned. 100 = 1 cent."
        ),
        kind="cents",
        minimum=_d("0"),
        maximum=_d("5000"),
        step=_d("25"),
    ),
    # -- how much may be at stake -------------------------------------------
    Tunable(
        key="event.max_position_pct_equity",
        domain="event",
        field="max_position_pct_equity",
        label="Largest position",
        description=(
            "The most of the account one market may hold, as a fraction of "
            "equity. 0.10 = 10%. This is the main concentration control: at 0.25 "
            "four bad markets can account for the whole balance."
        ),
        kind="percent",
        minimum=_d("0.001"),
        maximum=_d("0.25"),
        step=_d("0.005"),
    ),
    Tunable(
        key="event.max_portfolio_exposure_pct",
        domain="event",
        field="max_portfolio_exposure_pct",
        label="Total exposure",
        description=(
            "The most of the account that may be deployed across every open "
            "position at once. 0.60 = 60%. The remainder is the buffer that "
            "absorbs adverse moves without forcing an exit."
        ),
        kind="percent",
        minimum=_d("0.01"),
        maximum=_d("0.95"),
        step=_d("0.05"),
    ),
    Tunable(
        key="event.max_order_notional_usd",
        domain="event",
        field="max_order_notional_usd",
        label="Largest single order",
        description=(
            "A hard ceiling in dollars on any one order, independent of the "
            "percentage limits. It is the backstop for an equity figure that is "
            "wrong: a percentage of a bad number is still a bad number."
        ),
        kind="usd",
        minimum=_d("0.01"),
        maximum=_d("10000"),
        step=_d("5"),
    ),
    # -- when to stop --------------------------------------------------------
    Tunable(
        key="event.max_loss_per_trade_pct",
        domain="event",
        field="max_loss_per_trade_pct",
        label="Loss limit per trade",
        description=(
            "The most a single trade may be allowed to lose, as a fraction of "
            "equity. 0.02 = 2%."
        ),
        kind="percent",
        minimum=_d("0.001"),
        maximum=_d("0.10"),
        step=_d("0.005"),
    ),
    Tunable(
        key="event.max_daily_loss_pct",
        domain="event",
        field="max_daily_loss_pct",
        label="Daily loss limit",
        description=(
            "Losses in one day beyond this halt new entries until the next "
            "session. 0.05 = 5%. This is the control that stops a bad day "
            "becoming a bad week."
        ),
        kind="percent",
        minimum=_d("0.005"),
        maximum=_d("0.25"),
        step=_d("0.005"),
    ),
    Tunable(
        key="event.max_drawdown_pct",
        domain="event",
        field="max_drawdown_pct",
        label="Maximum drawdown",
        description=(
            "How far below the account's high-water mark equity may fall before "
            "trading stops entirely. 0.20 = 20%."
        ),
        kind="percent",
        minimum=_d("0.01"),
        maximum=_d("0.50"),
        step=_d("0.01"),
    ),
    Tunable(
        key="event.max_open_positions",
        domain="event",
        field="max_open_positions",
        label="Open positions",
        description=(
            "How many markets may be held at once. More positions spread risk "
            "but also spread attention, and each one pays the spread twice."
        ),
        kind="count",
        minimum=_d("1"),
        maximum=_d("500"),
        step=_d("1"),
    ),
    Tunable(
        key="event.max_orders_per_minute",
        domain="event",
        field="max_orders_per_minute",
        label="Order rate limit",
        description=(
            "A ceiling on orders per minute. It bounds the damage a malfunction "
            "can do before anyone notices it."
        ),
        kind="count",
        minimum=_d("1"),
        maximum=_d("600"),
        step=_d("5"),
    ),
    # -- perpetuals ----------------------------------------------------------
    Tunable(
        key="perp.max_exposure_pct_equity",
        domain="perp",
        field="max_exposure_pct_equity",
        label="Total perpetual exposure",
        description=(
            "The most of the account that may be deployed across perpetual "
            "positions. Independent of, and always subordinate to, the 1.00x "
            "leverage ceiling: this can only make the platform more "
            "conservative, never less."
        ),
        kind="percent",
        minimum=_d("0.01"),
        maximum=_d("1.00"),
        step=_d("0.05"),
    ),
    Tunable(
        key="perp.max_position_pct_equity",
        domain="perp",
        field="max_position_pct_equity",
        label="Largest perpetual position",
        description="The most of the account one perpetual market may hold.",
        kind="percent",
        minimum=_d("0.001"),
        maximum=_d("0.50"),
        step=_d("0.005"),
    ),
    Tunable(
        key="perp.max_spread_cc",
        domain="perp",
        field="max_spread_cc",
        label="Widest perpetual spread",
        description="How far apart bid and ask may be on a perpetual before it is skipped.",
        kind="cents",
        minimum=_d("1"),
        maximum=_d("5000"),
        step=_d("25"),
    ),
    Tunable(
        key="perp.max_daily_loss_pct",
        domain="perp",
        field="max_daily_loss_pct",
        label="Perpetual daily loss limit",
        description="Losses on perpetuals in one day beyond this halt new entries.",
        kind="percent",
        minimum=_d("0.005"),
        maximum=_d("0.25"),
        step=_d("0.005"),
    ),
)

_BY_KEY = {t.key: t for t in TUNABLES}


def tunable_by_key(key: str) -> Tunable:
    tunable = _BY_KEY.get(key)
    if tunable is None:
        raise TunableError(
            f"{key!r} is not an adjustable risk parameter. "
            f"Adjustable keys are: {', '.join(sorted(_BY_KEY))}"
        )
    return tunable


def locked_parameters() -> list[dict[str, Any]]:
    """Risk properties that are shown but cannot be changed from anywhere.

    Returned alongside the tunables so the dashboard renders them in the same
    list, visibly inert. The honest framing is not "this slider is disabled" but
    "there is no slider, because there is no number to set" -- the enforcement
    is a comparison between an exposure and an equity, and it has no limit
    parameter to read.
    """
    return [
        {
            "key": "perp.max_leverage",
            "domain": "perp",
            "label": "Maximum perpetual leverage",
            "description": (
                "Projected exposure may never exceed account equity. This is not "
                "a setting with a conservative default -- it is a comparison "
                "between two integers in kat.risk.invariants with no limit "
                "parameter, verified by behavioural canaries before every "
                "enforcement call. No configuration file, dashboard control, "
                "strategy, model or recovery path can raise it."
            ),
            "kind": "leverage",
            "value": str(MAX_PERP_LEVERAGE),
            "minimum": str(MAX_PERP_LEVERAGE),
            "maximum": str(MAX_PERP_LEVERAGE),
            "step": "0",
            "default": str(MAX_PERP_LEVERAGE),
            "higher_is_riskier": True,
            "locked": True,
        }
    ]


def current_values(settings: AppSettings) -> list[dict[str, Any]]:
    """Every tunable with its current value, plus the locked entries."""
    defaults = AppSettings()
    return [t.to_dict(settings, defaults) for t in TUNABLES] + locked_parameters()


def apply_override(settings: AppSettings, key: str, value: Any) -> tuple[Decimal, Decimal]:
    """Set one tunable on a live settings object.

    Returns ``(previous, applied)``. Raises :class:`TunableError` for an unknown
    key or an out-of-range value; the caller is expected to surface the message
    rather than swallow it.

    The value is written and then read back. Pydantic validators clamp on
    assignment, so what was requested and what took effect are not always the
    same number, and the caller needs the one that actually took effect for the
    audit record.
    """
    tunable = tunable_by_key(key)
    coerced = tunable.coerce(value)
    block = tunable.settings_block(settings)
    previous = tunable.read(settings)
    setattr(block, tunable.field, coerced)
    applied = tunable.read(settings)
    return previous, applied


def domain_of(key: str) -> MarketType:
    return MarketType.EVENT if tunable_by_key(key).domain == "event" else MarketType.PERP
