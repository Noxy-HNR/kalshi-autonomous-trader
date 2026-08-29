"""The documentation model, served to the dashboard.

One source for every explanation the interface shows. The hover tooltips, the
"configured limits" table and the docs page all read from here, so a term is
worded the same way wherever it appears and cannot drift between them.

Written for someone who has never traded. Three rules:

1. **No term is assumed.** Spread, fill, edge, drawdown, notional, settlement,
   calibration, leverage and arbitrage are each explained where they are used.
2. **Say what it costs them, not what it is.** "Open interest is the number of
   outstanding contracts" is a definition. "Nobody may be there to buy when you
   want out" is the thing worth knowing.
3. **Every setting is a trade-off, stated fairly.** None of these has a right
   answer, and a page that only lists the danger of one direction is arguing
   rather than explaining.

Anchors are stable: the dashboard links to ``#<anchor>`` from the control that
the entry describes, so renaming one breaks a link that a person is using.
"""

from __future__ import annotations

from typing import Any

from kat.config.settings import AppSettings
from kat.risk.tunables import current_values

__all__ = ["CONCEPTS", "CONTROLS", "RISK_FIELDS", "build_docs"]


#: Vocabulary. Every one of these appears somewhere in the interface.
CONCEPTS: list[dict[str, str]] = [
    {
        "anchor": "concept-event-contract",
        "term": "Event contract",
        "short": "A yes-or-no bet on something real, paying $1 or $0.",
        "body": (
            "An event contract asks a question with two possible answers, such as "
            "\"Will the Federal Reserve cut interest rates in September?\". You buy "
            "either the YES side or the NO side. When the answer is known, the "
            "winning side pays exactly $1.00 per contract and the losing side pays "
            "nothing.\n\n"
            "The price is the market's collective guess at the odds. A contract "
            "trading at 40 cents means the market thinks there is roughly a 40 "
            "percent chance. If you buy YES at 40 cents and it happens, you receive "
            "$1.00 for something you paid 40 cents for, a profit of 60 cents. If it "
            "does not happen, you lose the 40 cents you paid.\n\n"
            "The important safety property is that the most you can lose is exactly "
            "what you paid. There is no borrowing and no way to end up owing money."
        ),
    },
    {
        "anchor": "concept-perpetual",
        "term": "Perpetual future",
        "short": "A position that tracks a price and never expires.",
        "body": (
            "A perpetual future tracks the price of something and, unlike an event "
            "contract, never resolves. There is no date at which you find out "
            "whether you were right; the position simply gains or loses value as "
            "the price moves, for as long as you hold it.\n\n"
            "Perpetuals normally allow leverage, which means trading with borrowed "
            "money. This platform will not do that. See Leverage below.\n\n"
            "They are the riskier of the two products even without leverage, "
            "because a losing position has no natural end. An event contract "
            "resolves and the matter is settled; a perpetual can drift against you "
            "indefinitely."
        ),
    },
    {
        "anchor": "concept-leverage",
        "term": "Leverage",
        "short": "Trading with borrowed money. Capped here at 1x, permanently.",
        "body": (
            "Leverage means controlling a larger position than your money would "
            "otherwise allow, by borrowing the difference. At 2x leverage, $100 of "
            "your own money controls $200 of position. That doubles gains and it "
            "doubles losses: a 50 percent move against you does not halve your "
            "account, it empties it.\n\n"
            "This platform is capped at 1x. It can only ever commit money the "
            "account actually holds. A position can lose value, but it cannot leave "
            "you owing more than you put in.\n\n"
            "Unlike every other limit here, this one is not a setting. There is no "
            "number to change and no slider anywhere. The check is a comparison "
            "between what a trade would cost and what the account holds, and it "
            "refuses anything larger. No configuration file, dashboard control, "
            "strategy, model or recovery path can raise it."
        ),
    },
    {
        "anchor": "concept-spread",
        "term": "Spread",
        "short": "The gap between the buying and selling price. You pay it twice.",
        "body": (
            "Every market has two prices at once. The bid is the highest price "
            "anyone is currently willing to pay. The ask is the lowest price anyone "
            "is willing to accept. The gap between them is the spread.\n\n"
            "You buy at the ask and sell at the bid, so you pay the spread on the "
            "way in and again on the way out. On a market quoted 40 cents bid and "
            "43 cents ask, buying and immediately selling loses you 3 cents per "
            "contract without anything having happened.\n\n"
            "This is the single largest cost most strategies face, larger than the "
            "exchange fee. It is why the platform refuses to trade markets quoted "
            "too widely, and why loosening that limit does not make it trade better "
            "so much as make it willing to trade worse."
        ),
    },
    {
        "anchor": "concept-depth",
        "term": "Depth",
        "short": "How many contracts are waiting at each price.",
        "body": (
            "Depth is how many contracts other people currently have orders in for. "
            "A market can show an attractive price with almost nothing behind it.\n\n"
            "Thin depth usually lets you in but not out. You can buy at a fair price "
            "because someone is selling, and then find nobody willing to buy when you "
            "want to leave. A position you cannot close is a position that decides "
            "its own outcome.\n\n"
            "It is measured in contracts. The platform will not enter a market with "
            "less depth than the minimum you set."
        ),
    },
    {
        "anchor": "concept-edge",
        "term": "Edge",
        "short": "How mispriced the platform thinks a market is.",
        "body": (
            "The platform estimates what a contract is really worth, then compares "
            "that to the price being asked. The difference is the edge.\n\n"
            "If it thinks a contract is worth 45 cents and can buy it for 41, the "
            "edge is 4 cents. That is the entire reason to trade: without an edge, "
            "buying and selling just pays the spread and the fee.\n\n"
            "The estimate is itself uncertain, which is why a minimum edge exists. "
            "A 1 cent edge is easily swallowed by being 1 cent wrong about fair "
            "value, so acting on small edges produces many trades that are not "
            "actually profitable."
        ),
    },
    {
        "anchor": "concept-fill",
        "term": "Fill and slippage",
        "short": "An order is a request. A fill is it actually happening.",
        "body": (
            "Submitting an order does not mean you bought anything. The order goes "
            "to the exchange and may be filled completely, filled partially, or not "
            "at all.\n\n"
            "Slippage is the difference between the price you expected and the price "
            "you got. Between deciding and filling, the market moves. Slippage "
            "tolerance is how much of that the platform will accept before giving up "
            "on the order rather than chasing a price that is running away."
        ),
    },
    {
        "anchor": "concept-drawdown",
        "term": "Drawdown",
        "short": "How far the account has fallen from its best-ever value.",
        "body": (
            "Drawdown measures decline from a peak, not from where you started. If "
            "the account grew to $250 and is now $200, that is a 20 percent "
            "drawdown, even though you are still up on your original $200.\n\n"
            "It matters more than total profit for judging whether something is "
            "working. A strategy that makes money but repeatedly falls 40 percent "
            "along the way is one you will abandon at the worst possible moment."
        ),
    },
    {
        "anchor": "concept-fees",
        "term": "Fees",
        "short": "Kalshi charges most where the price is nearest 50 cents.",
        "body": (
            "Kalshi's trading fee is not a flat percentage. It is largest for "
            "contracts priced near 50 cents and smallest near the extremes.\n\n"
            "A round trip costs roughly 3.5 cents per contract at 50 cents, and "
            "about 1.3 cents at 10 or 90 cents. That shape drives a great deal of "
            "the platform's behaviour: a 2 cent edge is worth taking on a 10 cent "
            "contract and is not worth taking on a 50 cent one, because the fee "
            "alone exceeds it."
        ),
    },
    {
        "anchor": "concept-paper",
        "term": "Paper, demo and live",
        "short": "Three modes. Only one can move real money.",
        "body": (
            "In paper mode the platform watches real prices and simulates what would "
            "have happened, without sending a single order anywhere. Nothing is "
            "real, including the profit.\n\n"
            "In demo mode it sends genuine orders to Kalshi's practice exchange, "
            "which uses fake money.\n\n"
            "In live mode it sends real orders with real money. That requires two "
            "separate settings to be changed deliberately, and setting only one of "
            "them stops the platform from starting at all rather than running in a "
            "state nobody chose."
        ),
    },
    {
        "anchor": "concept-overfitting",
        "term": "Overfitting",
        "short": "A strategy that memorised the past instead of learning from it.",
        "body": (
            "Overfitting is the most common way automated trading fails. A strategy "
            "is tuned until it performs beautifully on historical data, because it "
            "has accidentally memorised that data's quirks rather than found "
            "anything real. It then loses money on everything new.\n\n"
            "This is why results are kept strictly separate by stage and never "
            "mixed. A strategy that did well on the data it was built from has "
            "demonstrated nothing at all. It has to keep working on data it has "
            "never seen, then on live prices with fake money, before it is allowed "
            "anywhere near real money."
        ),
    },
    {
        "anchor": "concept-calibration",
        "term": "Calibration",
        "short": "Whether a model's confidence matches reality.",
        "body": (
            "A calibrated model that says \"70 percent\" is right about 70 percent "
            "of the time. That matters more than raw accuracy: a model that is "
            "confidently wrong is far more dangerous than one that admits "
            "uncertainty, because position sizes follow confidence.\n\n"
            "A model never places an order here. It contributes one opinion among "
            "several, and every risk check runs regardless of what it says."
        ),
    },
    {
        "anchor": "concept-arbitrage",
        "term": "Arbitrage",
        "short": "Prices that contradict each other.",
        "body": (
            "If YES costs 40 cents and NO costs 45 cents on the same question, "
            "buying both costs 85 cents and is guaranteed to return exactly $1.00, "
            "because exactly one of them must happen. That 15 cent difference is "
            "arbitrage.\n\n"
            "Genuine cases are rare and small, and fees consume most of them. "
            "Everything the platform reports is priced after fees and after checking "
            "there are enough contracts actually available on every leg. An "
            "arbitrage where one leg fails to fill is not an arbitrage; it is an "
            "accidental bet."
        ),
    },
    {
        "anchor": "concept-point-in-time",
        "term": "Point-in-time integrity",
        "short": "Never testing with information you would not have had.",
        "body": (
            "When testing a strategy against history, it is dangerously easy to let "
            "it see something it could not have known yet. The result looks "
            "excellent and is worthless.\n\n"
            "Everything ingested here carries two timestamps: when the source says "
            "it happened, and when this platform actually received it. Only the "
            "second is ever used for a decision. A news story published at noon but "
            "received at 12:05 cannot influence a simulated trade at 12:01."
        ),
    },
    {
        "anchor": "concept-equity",
        "term": "Equity",
        "short": "What the account is worth right now, if everything were closed.",
        "body": (
            "Equity is your cash plus what your open positions are currently worth. "
            "It is the number every limit on this platform is measured against: a "
            "rule saying one position may use 10 percent means 10 percent of this.\n\n"
            "It moves for two different reasons, and the difference matters. Cash "
            "changes when a trade settles or a fee is charged, and that is final. "
            "Position value changes as prices move, and that is not final until you "
            "close -- a position showing a gain has not made you money yet.\n\n"
            "Event contracts and perpetuals hold their money in separate places on "
            "Kalshi, so this platform tracks two equity figures and shows the total. "
            "Moving money between them is something only you can do, from Kalshi's "
            "own site; the platform has no ability to do it."
        ),
    },
    {
        "anchor": "concept-exposure",
        "term": "Exposure",
        "short": "How much money is currently at risk, rather than sitting idle.",
        "body": (
            "Exposure is the total value of everything you currently hold. Equity is "
            "what you have; exposure is how much of it is committed to positions "
            "rather than waiting as cash.\n\n"
            "Zero exposure means every dollar is sitting in cash and nothing can be "
            "lost to a price move. Exposure equal to your equity means every dollar "
            "is committed. The platform will never go beyond that on perpetuals -- "
            "that is the 1x leverage cap, and it is the one limit with no setting.\n\n"
            "High exposure is not the same as high risk if the positions are "
            "unrelated, and it is not much better than one large bet if they are "
            "not. That is what the correlated-exposure limit is for."
        ),
    },
    {
        "anchor": "concept-pnl",
        "term": "Realised and unrealised P&L",
        "short": "Money you have actually made, versus money you might still lose.",
        "body": (
            "P&L is profit and loss. It comes in two kinds and they are not "
            "interchangeable.\n\n"
            "Realised P&L is from positions you have closed. The trade is finished, "
            "the money has moved, and nothing can take it back. This is the number "
            "that tells you whether the platform is actually working.\n\n"
            "Unrealised P&L is from positions you still hold, valued at what they "
            "would fetch right now. It is a genuine estimate and it is not yours "
            "yet: a position up 30 cents can be down 30 cents an hour later, and "
            "the only way to convert one into the other is to close the position "
            "and pay the spread and the fee to do it.\n\n"
            "Daily-loss limits count realised losses, because a paper loss that "
            "recovers by evening was never a loss. A drawdown limit watches equity "
            "itself, which includes both."
        ),
    },
    {
        "anchor": "concept-position",
        "term": "Position",
        "short": "Contracts you currently hold and have not yet closed.",
        "body": (
            "A position is what you are left holding after a trade. Holding 20 YES "
            "contracts on one market is one position of 20 contracts.\n\n"
            "A position ends in one of two ways. You close it by selling, which "
            "requires somebody willing to buy -- this is why the platform refuses "
            "markets with too little depth, because a position you cannot close is "
            "one that decides its own outcome. Or, for an event contract, it "
            "settles: the question is answered and each contract pays $1.00 or "
            "nothing, with no trading required.\n\n"
            "Perpetuals never settle. They only end when you close them, which is "
            "why they are the riskier of the two even at 1x leverage."
        ),
    },
    {
        "anchor": "concept-allocation",
        "term": "Allocation",
        "short": "How the account is split between domains and strategies.",
        "body": (
            "Allocation is how the money is divided up: how much sits behind event "
            "contracts versus perpetuals, and how much each strategy may use.\n\n"
            "The reason to split it at all is that concentration is the fastest way "
            "to lose an account. A strategy given everything will occasionally be "
            "confidently wrong with everything.\n\n"
            "A strategy with nothing allocated to it is not disabled: it falls back "
            "to its own configured ceiling, which is how a strategy is usable before "
            "any allocation has been decided. On a small account these ceilings, not "
            "the allocation, are usually what actually binds."
        ),
    },
]


#: Risk settings shown in the "configured limits" table that are not sliders.
#: The tunables carry their own text; these are the remainder, so every row in
#: that table has an explanation rather than only the adjustable ones.
RISK_FIELDS: list[dict[str, str]] = [
    {
        "anchor": "field-max_correlated_exposure_pct",
        "field": "max_correlated_exposure_pct",
        "term": "Maximum correlated exposure",
        "short": "How much can ride on outcomes that tend to move together.",
        "body": (
            "Ten positions are only ten separate bets if they can fail "
            "independently. Ten contracts on related questions are closer to one "
            "large bet wearing a disguise, and they will lose together.\n\n"
            "This caps how much of the account can sit in markets that move as a "
            "group, which is a limit the per-market position cap cannot express."
        ),
    },
    {
        "anchor": "field-min_expected_value_micros",
        "field": "min_expected_value_micros",
        "term": "Minimum expected value",
        "short": "The smallest profit worth the trouble, after all costs.",
        "body": (
            "After the spread, the fee and the estimated slippage, a trade has some "
            "expected profit left. This is the floor beneath which it is not worth "
            "doing.\n\n"
            "It exists because a trade can have a genuine edge and still not be "
            "worth taking. Making a fraction of a cent per contract is not a "
            "strategy, it is a way to pay fees."
        ),
    },
    {
        "anchor": "field-min_liquidation_buffer_pct",
        "field": "min_liquidation_buffer_pct",
        "term": "Minimum liquidation buffer",
        "short": "Distance kept from the price that would close you out.",
        "body": (
            "On perpetuals, if a position loses enough the exchange closes it for "
            "you, at whatever price it can get. That is liquidation, and it is the "
            "worst possible way to exit.\n\n"
            "At 1x leverage liquidation should be unreachable, since you can never "
            "lose more than you put in. This buffer is verified rather than assumed, "
            "because \"should be unreachable\" is not the same as \"is\"."
        ),
    },
    {
        "anchor": "field-max_funding_rate_bps_per_hour",
        "field": "max_funding_rate_bps_per_hour",
        "term": "Maximum funding rate",
        "short": "The most you will pay per hour to hold a perpetual.",
        "body": (
            "Perpetuals charge a recurring payment between the two sides, called "
            "funding. If you hold the crowded side, you pay it; if you hold the "
            "unpopular side, you receive it.\n\n"
            "It is a real cost of simply holding a position, independent of whether "
            "the price moves. This caps how expensive that is allowed to get before "
            "the platform declines to hold the position at all."
        ),
    },
]


#: Controls, as opposed to numbers.
CONTROLS: list[dict[str, str]] = [
    {
        "anchor": "control-kill-switch",
        "term": "Kill switches",
        "short": "Stop trading, and stay stopped.",
        "body": (
            "A kill switch halts trading immediately and remains engaged through a "
            "restart. That last part is the point: a halt you can undo by turning "
            "the process off and on again is not a halt.\n\n"
            "There are separate switches per area, so you can stop event-contract "
            "trading while leaving perpetuals running, or stop everything at once. "
            "One is engaged automatically if reconciliation cannot confirm what "
            "positions you actually hold.\n\n"
            "Engaging one requires a written reason, which is recorded in the audit "
            "log alongside who did it. Releasing one is also recorded."
        ),
    },
    {
        "anchor": "control-operating-mode",
        "term": "Operating mode",
        "short": "How much the platform may do without asking you.",
        "body": (
            "Four modes, differing only in who acts on a passing result:\n\n"
            "Gated discovery, the default and the recommended one, lets the platform "
            "research and validate strategies on its own but applies no promotion "
            "without your approval.\n\n"
            "Autonomous discovery applies promotions up to paper trading, and asks "
            "before anything reaches real money.\n\n"
            "Full autonomy applies promotions itself, including into live capital.\n\n"
            "Human strategies performs no promotions at all; it measures and reports.\n\n"
            "The tests a strategy must pass are identical in every mode. Autonomy "
            "widens what may be done on good evidence; it never changes what counts "
            "as good evidence. A strategy that fails its checks is never promoted in "
            "any mode, and a strategy that is losing money is demoted immediately in "
            "every mode, without waiting for approval."
        ),
    },
    {
        "anchor": "control-pause",
        "term": "Pause and resume",
        "short": "Stop opening new positions without halting everything.",
        "body": (
            "Pausing stops new entries while leaving existing positions managed. It "
            "is the gentler option when you want to stop adding risk but not abandon "
            "what is already open.\n\n"
            "A kill switch is the stronger action and survives a restart; a pause "
            "does not."
        ),
    },
    {
        "anchor": "control-cancel-all",
        "term": "Cancel all orders",
        "short": "Withdraw every order that has not yet filled.",
        "body": (
            "This cancels working orders. It does not close positions you already "
            "hold: an order is a request that has not happened yet, while a position "
            "is something you own.\n\n"
            "Useful when you want to stop what is about to happen without changing "
            "what already has."
        ),
    },
    {
        "anchor": "control-remote-access",
        "term": "Remote access",
        "short": "Opening this dashboard from your phone or another laptop.",
        "body": (
            "By default the dashboard answers only on the machine it is running "
            "on. Type its address into another device and nothing happens, "
            "because nothing outside this computer can reach it at all.\n\n"
            "That default exists because this dashboard has no password. It can "
            "stop trading, change how much money may be risked, and cancel every "
            "order. Putting it on an ordinary home or office network would mean "
            "anything else on that network could do those things too.\n\n"
            "The supported way to reach it from elsewhere is Tailscale, a free "
            "private network you install on your own devices. It gives each of "
            "your devices an address only your other devices can reach: there is "
            "no route to it from the internet, and traffic between them is "
            "encrypted. Set KAT_API_REMOTE_ACCESS=tailscale in your .env file, "
            "restart, and the dashboard answers on that address instead.\n\n"
            "To set it up: install Tailscale on this machine and on the phone or "
            "laptop you want to use, sign both into the same account, then set "
            "that line and restart. The System page will show the address to "
            "open, and the launcher prints it too.\n\n"
            "If Tailscale is not installed or not running, the platform still "
            "starts -- it just stays reachable from this machine only, and says "
            "so. It will never fall back to something more open. There is no "
            "setting that exposes it to everything, which is deliberate: that is "
            "the mistake this design exists to make impossible."
        ),
    },
    {
        "anchor": "control-tuning",
        "term": "Risk tuning",
        "short": "Adjust the limits. Changes are live, audited and permanent.",
        "body": (
            "Every slider changes a real limit on the running platform immediately. "
            "There is no apply button because there is no draft: what you see is "
            "what is in force.\n\n"
            "Each change is written to the audit log with the old value, the new "
            "value and who made it, and persisted so a restart does not quietly "
            "revert it. A limit that returns to its default overnight is worse than "
            "one never changed, because you would believe a bound was in force that "
            "was not.\n\n"
            "Drag to change. Double-click any slider to restore its default. "
            "Middle-click to open this page at that setting's entry. The small tick "
            "on each track marks where the default sits."
        ),
    },
]


def build_docs(settings: AppSettings | None = None) -> dict[str, Any]:
    """The whole documentation model, with live values for every limit.

    Values are included so the page shows what the platform is actually set to
    rather than what it shipped with. A docs page describing settings you do not
    have is worse than none.
    """
    tunables = current_values(settings or AppSettings())
    return {
        "concepts": CONCEPTS,
        "tunables": [
            {**t, "anchor": f"tunable-{t['key'].replace('.', '-')}"} for t in tunables
        ],
        "risk_fields": RISK_FIELDS,
        "controls": CONTROLS,
    }
