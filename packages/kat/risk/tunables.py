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
    #: The hover explainer, written for someone who has never traded. It may
    #: not assume any term: "spread", "edge", "drawdown" and "notional" all
    #: have to be explained where they are used, not looked up elsewhere.
    plain: str = ""
    #: What raising this buys you, and what it costs. Shown in the docs page as
    #: the case *for* a larger number, stated fairly -- every one of these is a
    #: real trade-off rather than a setting with a right answer.
    raising: str = ""
    #: The same for lowering it.
    lowering: str = ""
    #: A concrete worked example, in dollars and cents on a small account.
    example: str = ""
    #: Overrides the per-unit default in _STEP_BY_KIND. Left unset almost
    #: always: an inconsistent step is what made these feel arbitrary.
    step_override: Decimal | None = None
    #: True when a larger number means more risk. The dashboard colours the
    #: track by this, so an operator can see which way "riskier" is without
    #: reading each description.
    higher_is_riskier: bool = True

    @property
    def step(self) -> Decimal:
        return self.step_override or _STEP_BY_KIND.get(self.kind, Decimal("1"))

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
            "plain": self.plain,
            "raising": self.raising,
            "lowering": self.lowering,
            "example": self.example,
            "kind": self.kind,
            "value": str(self.read(settings)),
            "minimum": str(self.minimum),
            "maximum": str(self.maximum),
            "step": str(self.step),
            "higher_is_riskier": self.higher_is_riskier,
            "locked": False,
        }


#: Slider granularity, by unit rather than per parameter.
#:
#: Hand-picking a step per slider produced an inconsistent feel -- some moved in
#: half-percent jumps, others in five-percent ones, so dragging two sliders the
#: same physical distance changed the risk by very different amounts. These are
#: fine enough to land on a considered number and uniform enough that the
#: controls behave the same way as each other.
_STEP_BY_KIND: dict[str, Decimal] = {
    "percent": Decimal("0.001"),   # 0.1 of a percentage point
    "cents": Decimal("10"),        # 0.1 of a cent
    "contracts": Decimal("50"),    # half a contract
    "usd": Decimal("1"),
    "count": Decimal("1"),
    "micros": Decimal("1000"),
}


def _d(value: str) -> Decimal:
    return Decimal(value)


#: The adjustable set.
#:
#: Bounds are the *usable* range, deliberately narrower than the clamp in
#: kat.config.settings. A slider whose maximum is 500 open positions or 1000
#: contracts of depth is technically honest and practically useless: every
#: sensible value crowds into the left tenth of the track. Someone who genuinely
#: wants a value beyond these can set it in .env, where the clamp is still the
#: real limit and still applies.
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
        plain=(
            "Every market has two prices at once: what buyers will pay and what "
            "sellers will accept. The gap between them is the spread, and you pay it "
            "twice -- once buying in and once selling out. On a 3-cent spread you "
            "start about 3 cents behind on every trade and have to be right by more "
            "than that to make money. Raise this and the platform will trade markets "
            "where that starting hole is deeper. It does not make those trades "
            "better; it makes the platform willing to take them."
        ),
        raising=(
            "Lets the platform trade markets where the buying and selling prices sit "
            "further apart. You get access to far more markets, because most markets "
            "are not tightly priced. The cost is that you start every one of those "
            "trades further behind: the gap is paid twice, once getting in and once "
            "getting out, before any profit is possible."
        ),
        lowering=(
            "Restricts trading to tightly-priced markets only. Every trade starts "
            "closer to break-even and needs a smaller move to become profitable. The "
            "cost is that the platform will often find nothing worth trading and sit "
            "idle, which feels like it is broken but is usually the correct answer."
        ),
        example=(
            "At the 3 cent default, buying a contract at 40 cents means you could "
            "only sell it back for about 37 cents immediately. You are 3 cents down "
            "before anything happens. Raise this to 10 cents and you would be 10 "
            "cents down instead, needing a much bigger move just to break even."
        ),
        kind="cents",
        minimum=_d("50"),
        maximum=_d("2000"),
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
        plain=(
            "How many contracts other people currently have orders in for. If almost "
            "nobody is offering to trade, you can usually get IN at a fair price but "
            "not back OUT -- there is nobody on the other side when you want to sell. "
            "Lower this and the platform will enter markets it may struggle to leave."
        ),
        raising=(
            "Requires more contracts to be waiting before the platform will enter. "
            "Fewer markets qualify, but you are much more likely to be able to sell "
            "when you want to."
        ),
        lowering=(
            "Admits thinner markets where few people are trading. More opportunities, "
            "but you may not be able to get out at a sensible price. Being unable to "
            "exit is worse than never entering: a position you cannot close is one "
            "that decides its own outcome."
        ),
        example=(
            "At the 10 contract default, if you buy 5 contracts there is roughly "
            "enough interest for you to sell them back. Drop it to 1 contract and you "
            "might buy 5 and find only 1 buyer waiting, leaving you holding 4 you "
            "cannot sell."
        ),
        kind="contracts",
        minimum=_d("100"),
        maximum=_d("10000"),
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
        plain=(
            "The platform estimates what a contract is really worth, then compares "
            "that to the price being asked. The gap is the edge -- how mispriced it "
            "thinks the market is. This sets how big that gap must be before it acts. "
            "Lower it and it trades more often on smaller, less certain mispricings, "
            "where being slightly wrong wipes out the gain."
        ),
        raising=(
            "Demands a bigger mispricing before acting. Fewer trades, but each has "
            "more room for the platform's estimate of fair value to be somewhat wrong "
            "and still come out ahead."
        ),
        lowering=(
            "Acts on smaller mispricings. Many more trades, but each has a thinner "
            "margin. Since the estimate of fair value is itself uncertain, a small "
            "edge can easily be no edge at all once fees are paid."
        ),
        example=(
            "At the 2 cent default, if the platform thinks a contract is worth 45 "
            "cents it will only buy below 43. Lower it to 0.5 cents and it would buy "
            "at 44.5, where being half a cent wrong about fair value wipes out the "
            "entire trade."
        ),
        kind="cents",
        minimum=_d("20"),
        maximum=_d("1000"),
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
        plain=(
            "Between deciding to trade and the order actually filling, the price can "
            "move. Slippage is how much worse than expected you ended up paying. This "
            "is how much of that the platform will tolerate before giving up on the "
            "order instead of chasing a price that is running away."
        ),
        raising=(
            "Tolerates the price moving further against you between deciding and "
            "filling. More orders complete, but some complete at prices you would not "
            "have chosen."
        ),
        lowering=(
            "Abandons an order as soon as the price moves. You never overpay, but in "
            "a fast market you may get almost nothing filled, which can leave you "
            "half in a position you wanted fully."
        ),
        example=(
            "At the 1 cent default, an order to buy at 40 cents is abandoned if the "
            "price runs to 41 before it fills. That protects you from chasing, but if "
            "news is moving the market you will miss it."
        ),
        kind="cents",
        minimum=_d("0"),
        maximum=_d("500"),
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
        plain=(
            "How much of your account can ride on any single market. At 10%, one "
            "market going completely against you costs at most a tenth of the "
            "balance. At 25% you only need four bad ones to lose everything. This is "
            "the single most important control against being wrong about one thing in "
            "a big way."
        ),
        raising=(
            "Allows more of the account into a single market. When the platform is "
            "right it wins more; when it is wrong about one thing it loses much more. "
            "This is concentration, and concentration cuts both ways with perfect "
            "symmetry."
        ),
        lowering=(
            "Spreads the account across more markets, so no single one can hurt "
            "badly. The cost is that a genuinely good opportunity is capped, and each "
            "small position still pays the full spread both ways, so many tiny trades "
            "bleed fees."
        ),
        example=(
            "On a 200 dollar account at the 10 percent default, one market can hold "
            "20 dollars. Raise it to 25 percent and one market holds 50 dollars, so "
            "four bad markets could account for the whole balance."
        ),
        kind="percent",
        minimum=_d("0.001"),
        maximum=_d("0.25"),
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
        plain=(
            "How much of your account can be committed to trades all at once, across "
            "every market together. The rest stays as cash. That cash is not idle -- "
            "it is what lets positions move against you for a while without forcing "
            "you to sell at the worst moment."
        ),
        raising=(
            "Puts more of the account to work at once. More of your money is deployed "
            "rather than idle. The cost is a smaller cash buffer, so a broad move "
            "against you leaves less room before positions must be closed at bad "
            "prices."
        ),
        lowering=(
            "Keeps more of the account in cash. That cash is what lets positions move "
            "against you for a while without forcing a sale at the worst moment. The "
            "cost is that the account works less hard."
        ),
        example=(
            "On 200 dollars at the 60 percent default, up to 120 dollars is deployed "
            "and 80 stays as cash. At 95 percent only 10 dollars is left, so a modest "
            "adverse move across several positions leaves almost no cushion."
        ),
        kind="percent",
        minimum=_d("0.01"),
        maximum=_d("0.95"),
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
        plain=(
            "A hard dollar ceiling on any one order, no matter what the percentages "
            "allow. It exists for the case where the account balance is read wrong: a "
            "percentage of a wrong number is still a wrong number, and this catches "
            "it."
        ),
        raising=(
            "Permits larger individual orders. Useful as the account grows, since "
            "otherwise the percentage limits would be the only constraint."
        ),
        lowering=(
            "Caps any single order in plain dollars regardless of what the "
            "percentages allow. This is the backstop for the case where the account "
            "balance is misread: a percentage of a wrong number is still a wrong "
            "number, and this catches it."
        ),
        example=(
            "At the 25 dollar default, even if a bug reported your balance as one "
            "million dollars, no single order could exceed 25 dollars."
        ),
        kind="usd",
        minimum=_d("1"),
        maximum=_d("500"),
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
        plain=(
            "How much of the account a single trade is allowed to lose before it is "
            "closed. Think of it as the size of the individual mistakes you are "
            "willing to make."
        ),
        raising=(
            "Gives each trade more room to go against you before being closed. Fewer "
            "positions are cut early and then recover without you. The cost is that "
            "each individual mistake is larger."
        ),
        lowering=(
            "Cuts losing trades sooner. Individual mistakes stay small, but positions "
            "that would have recovered get closed at the bottom. Being stopped out "
            "repeatedly in a choppy market is its own way to lose money."
        ),
        example=(
            "On 200 dollars at the 2 percent default, a trade is closed once it is "
            "down about 4 dollars. At 10 percent it would run to 20 dollars before "
            "being cut."
        ),
        kind="percent",
        minimum=_d("0.001"),
        maximum=_d("0.10"),
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
        plain=(
            "If losses reach this much in one day, the platform stops opening new "
            "trades until tomorrow. This is what stops a bad morning turning into a "
            "bad month: the most common way automated systems fail is trading harder "
            "to win losses back."
        ),
        raising=(
            "Allows a worse day before trading stops. The platform can trade through "
            "a rough patch instead of standing down. The cost is that a genuinely bad "
            "day runs further."
        ),
        lowering=(
            "Stops trading sooner after losses. This is the control that prevents a "
            "bad morning becoming a bad month; the most common way automated systems "
            "fail is trading harder to win losses back. The cost is halting on days "
            "that were merely unlucky."
        ),
        example=(
            "On 200 dollars at the 5 percent default, losing about 10 dollars in a "
            "day stops new trades until tomorrow. Existing positions are still "
            "managed; only new entries stop."
        ),
        kind="percent",
        minimum=_d("0.005"),
        maximum=_d("0.25"),
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
        plain=(
            "Drawdown is how far the account has fallen from its best-ever value. If "
            "it was $500 at its peak and is $400 now, that is a 20% drawdown. Reach "
            "this number and the platform stops trading entirely and waits for you. "
            "It is the final backstop, not a routine limit."
        ),
        raising=(
            "Permits the account to fall further from its best-ever value before "
            "halting entirely. Survives a deeper slump without intervention. The cost "
            "is that if something is genuinely broken, it breaks further before "
            "stopping."
        ),
        lowering=(
            "Halts sooner after a sustained decline. This is the final backstop, not "
            "a routine limit: reaching it means something went wrong that the daily "
            "limit did not catch. Setting it tight risks halting on ordinary "
            "variance."
        ),
        example=(
            "If the account peaked at 250 dollars, the 20 percent default halts "
            "everything at 200. It does not resume on its own; a human decides "
            "whether the strategy is broken or merely unlucky."
        ),
        kind="percent",
        minimum=_d("0.01"),
        maximum=_d("0.50"),
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
        plain=(
            "How many markets can be held at the same time. Spreading across more "
            "markets means no single one can hurt much -- but each one costs the "
            "spread to enter and exit, so holding many small positions can bleed "
            "money in fees alone."
        ),
        raising=(
            "Holds more markets at once. Risk is spread more thinly and more "
            "opportunities are taken. The cost is that each position pays the spread "
            "twice, so many small ones bleed fees, and more positions means less "
            "attention on each."
        ),
        lowering=(
            "Concentrates into fewer markets. Lower total fees and easier to follow. "
            "The cost is that each position matters more, and a single bad one is a "
            "bigger share of the outcome."
        ),
        example=(
            "At the 20 default, with a 10 percent position cap, the platform could in "
            "principle hold 20 markets at 10 percent each, except the total exposure "
            "limit binds first. That is exactly why several limits exist rather than "
            "one."
        ),
        kind="count",
        minimum=_d("1"),
        maximum=_d("100"),
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
        plain=(
            "A speed limit on orders. Not a trading strategy -- a safety catch. If "
            "something malfunctions, this bounds how much damage it can do in the "
            "minutes before anyone notices."
        ),
        raising=(
            "Lets the platform act faster when many opportunities appear at once. The "
            "cost is that a malfunction can do more damage before anyone notices."
        ),
        lowering=(
            "Throttles order flow. This is a safety catch rather than a strategy: it "
            "bounds how much harm a runaway loop can do in the minutes before a human "
            "intervenes. Set too low, genuine opportunities are missed while the "
            "platform waits its turn."
        ),
        example=(
            "At the 30 default, a bug submitting orders in a tight loop could place "
            "at most 30 in a minute rather than thousands. Combined with the 25 "
            "dollar order cap, worst-case damage per minute is bounded at 750 dollars "
            "of notional."
        ),
        kind="count",
        minimum=_d("1"),
        maximum=_d("120"),
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
        plain=(
            "How much of the account can be committed to perpetual futures at once. "
            "Separate from the 1.00x leverage ceiling and always weaker than it: this "
            "can only make the platform more cautious, never less."
        ),
        raising=(
            "Commits more of the account to perpetual futures. This can never exceed "
            "the 1x leverage ceiling: it only ever makes the platform more cautious "
            "than that hard limit, never less."
        ),
        lowering=(
            "Keeps perpetual exposure small. Perpetuals are the riskier of the two "
            "products even at 1x, because they do not resolve to a fixed outcome the "
            "way an event contract does."
        ),
        example=(
            "On 200 dollars at the 50 percent default, at most 100 dollars is "
            "committed to perpetuals. Even at 100 percent the leverage invariant "
            "still caps exposure at the account's equity."
        ),
        kind="percent",
        minimum=_d("0.01"),
        maximum=_d("1.00"),
    ),
    Tunable(
        key="perp.max_position_pct_equity",
        domain="perp",
        field="max_position_pct_equity",
        label="Largest perpetual position",
        description="The most of the account one perpetual market may hold.",
        plain=(
            "How much of the account can ride on a single perpetual market. Same idea "
            "as the event-contract position limit, applied to the other domain."
        ),
        raising=(
            "Allows more of the account into one perpetual market. Same trade-off as "
            "the event-contract position limit: more upside when right, more damage "
            "when wrong."
        ),
        lowering=(
            "Limits how much any single perpetual can cost you. Perpetual prices move "
            "continuously with no resolution date, so a position can drift against "
            "you indefinitely."
        ),
        example=(
            "On 200 dollars at the 15 percent default, one perpetual market can hold "
            "30 dollars."
        ),
        kind="percent",
        minimum=_d("0.001"),
        maximum=_d("0.50"),
    ),
    Tunable(
        key="perp.max_spread_cc",
        domain="perp",
        field="max_spread_cc",
        label="Widest perpetual spread",
        description="How far apart bid and ask may be on a perpetual before it is skipped.",
        plain=(
            "The gap between the buying and selling price on a perpetual, which you "
            "pay on the way in and again on the way out. Same meaning as the "
            "event-contract spread limit."
        ),
        raising=(
            "Trades perpetuals whose buying and selling prices sit further apart. "
            "Same cost as the event spread limit: the gap is paid twice."
        ),
        lowering=(
            "Restricts perpetual trading to tightly-priced markets. Fewer trades, "
            "each starting closer to break-even."
        ),
        example=(
            "At the 3 cent default the platform skips any perpetual quoted more than "
            "3 cents wide."
        ),
        kind="cents",
        minimum=_d("50"),
        maximum=_d("2000"),
    ),
    Tunable(
        key="perp.max_daily_loss_pct",
        domain="perp",
        field="max_daily_loss_pct",
        label="Perpetual daily loss limit",
        description="Losses on perpetuals in one day beyond this halt new entries.",
        plain=(
            "If perpetual trading loses this much in one day, it stops opening new "
            "positions until tomorrow. Counted separately from event contracts, so a "
            "bad day in one domain does not halt the other."
        ),
        raising=(
            "Allows a worse day on perpetuals before new entries stop. Counted "
            "separately from event contracts, so one domain having a bad day does not "
            "halt the other."
        ),
        lowering=(
            "Stops perpetual trading sooner after losses. Because perpetuals never "
            "resolve, a losing position has no natural end date, so stopping early "
            "matters more here than for event contracts."
        ),
        example=(
            "On 200 dollars at the 4 percent default, losing about 8 dollars on "
            "perpetuals in a day stops new perpetual entries. Event-contract trading "
            "continues unaffected."
        ),
        kind="percent",
        minimum=_d("0.005"),
        maximum=_d("0.25"),
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
            "plain": (
                "Leverage means trading with borrowed money. At 2x you control "
                "$200 of position with $100 of your own, so a 50% move against "
                "you does not halve your account -- it empties it. At 1x you can "
                "only ever put up money you actually have, so a position can "
                "lose value but cannot leave you owing more than you put in. "
                "This platform will not exceed 1x, and unlike everything else on "
                "this page there is no number here to change: the check is a "
                "comparison between what a trade would cost and what the account "
                "holds, and it refuses anything larger. No setting, strategy or "
                "model can reach past it."
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
