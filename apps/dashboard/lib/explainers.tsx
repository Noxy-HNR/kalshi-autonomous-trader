/**
 * Plain-language explainers for every page.
 *
 * Written for someone who has never traded. Two rules followed throughout:
 *
 * 1. No term is assumed. "Spread", "fill", "drawdown", "notional", "edge",
 *    "settlement", "arbitrage" and "calibration" are explained where they
 *    appear rather than left for the reader to look up.
 * 2. Say what it means for *them*, not what it is in the abstract. "Open
 *    interest is the number of outstanding contracts" is a definition; "nobody
 *    may be there to buy when you want out" is what it costs you.
 *
 * They live here rather than inline so the same concept is worded the same way
 * everywhere it appears, and so a page file stays about rendering.
 */

import type { ReactNode } from "react";

export const PAGE: Record<string, ReactNode> = {
  overview: (
    <>
      <strong>The whole platform at a glance.</strong> This is an automated trading
      program: it watches prices on Kalshi, decides whether anything looks mispriced, and
      places orders on its own.
      <br />
      <br />
      Two numbers matter most. <em>Equity</em> is what the account is worth right now.{" "}
      <em>Exposure</em> is how much of it is committed to open trades — money that is in
      the market and can move against you.
      <br />
      <br />
      In <strong>paper</strong> mode none of it is real: orders are simulated against live
      prices and no money moves.
    </>
  ),

  portfolio: (
    <>
      <strong>A visual read on the same numbers as Overview.</strong> Equity is what the
      account is worth; exposure is how much of it is committed to open trades right now.
      <br />
      <br />
      The equity-over-time chart is recorded locally in your browser from live polls, not
      pulled from a server-side history the engine doesn&rsquo;t keep &mdash; it starts over
      on a new device and grows the longer this page stays open.
    </>
  ),

  events: (
    <>
      <strong>Event contracts are yes-or-no bets on real events.</strong> &ldquo;Will the
      Fed cut rates in September?&rdquo; trades as a contract that pays exactly $1 if it
      happens and $0 if it does not.
      <br />
      <br />
      The price is the market&rsquo;s estimate of the odds: 40¢ means roughly a 40%
      chance. Buy at 40¢ and if it happens you receive $1 — a 60¢ profit. If it does not,
      you lose the 40¢ you paid.
      <br />
      <br />
      That is the key safety property: <strong>the most you can lose is what you paid.</strong>{" "}
      There is no borrowing and no way to owe more than you put in.
    </>
  ),

  perps: (
    <>
      <strong>Perpetual futures are the risky cousin of event contracts.</strong> They
      track a price, never expire, and normally allow <em>leverage</em> — trading with
      borrowed money, where losses can exceed your deposit.
      <br />
      <br />
      This platform will not do that. It is capped at <strong>1× leverage</strong>: it can
      only ever commit money the account actually holds. A position can lose value, but it
      cannot leave you owing anything.
      <br />
      <br />
      <em>Funding</em> is a small recurring payment between traders on each side. It is a
      real cost of holding a position and is counted against any expected profit.
    </>
  ),

  positions: (
    <>
      <strong>What the platform currently owns.</strong> Each row is a market it has
      bought into and not yet closed.
      <br />
      <br />
      <em>Unrealised</em> profit or loss is on paper — the position is still open and the
      number will keep moving. <em>Realised</em> is settled and final. A position that
      looks profitable is not money you have until it closes.
    </>
  ),

  orders: (
    <>
      <strong>Every order sent to the exchange, and what became of it.</strong> An order is
      a request, not a done deal — asking to buy does not mean you bought.
      <br />
      <br />
      A <em>fill</em> is an order that actually traded; <em>partial</em> means only some of
      it did. <em>Slippage</em> is how much worse the price was than expected by the time
      it filled, and <em>latency</em> is how long the exchange took to answer.
      <br />
      <br />
      Rejections are normal and usually protective: the platform refuses its own orders far
      more often than the exchange does.
    </>
  ),

  markets: (
    <>
      <strong>The markets being watched right now.</strong>
      <br />
      <br />
      <em>Bid</em> is the highest price anyone will currently pay; <em>ask</em> is the
      lowest anyone will accept. The gap between them is the <em>spread</em>, and you pay
      it twice — buying in and selling out. A wide spread means you start well behind.
      <br />
      <br />
      <em>Depth</em> is how many contracts are waiting at those prices. Thin depth means
      you may get in but struggle to get out. <em>Age</em> is how long since the exchange
      last told us anything; stale prices are not traded on.
    </>
  ),

  strategies: (
    <>
      <strong>A strategy is one idea about when something is mispriced.</strong> Each
      watches for a specific pattern and proposes trades when it sees it.
      <br />
      <br />
      None starts with real money. Each climbs a ladder — tested on history, then on
      history it has never seen, then on live prices with fake money — and only advances by
      producing evidence at each stage. Most never make it, which is the system working.
    </>
  ),

  discovery: (
    <>
      <strong>Strategies being tested, and how far each has got.</strong>
      <br />
      <br />
      The danger this guards against is <em>overfitting</em>: an idea that looks brilliant
      on past data because it accidentally memorised that data&rsquo;s quirks, then loses
      money on anything new. It is the most common way automated trading fails.
      <br />
      <br />
      So results are kept strictly separate by stage and never mixed. A strategy that did
      well on the data it was built from has proved nothing at all.
    </>
  ),

  models: (
    <>
      <strong>Machine-learning models that estimate probabilities.</strong> A model takes
      what it can measure about a market and outputs a number: how likely it thinks the
      answer is yes.
      <br />
      <br />
      <em>Calibration</em> matters more than accuracy. A well-calibrated model that says
      &ldquo;70%&rdquo; is right about 70% of the time. A model that is confident and wrong
      is far more dangerous than one that admits uncertainty.
      <br />
      <br />
      A model never places an order. It is one opinion among several, and every risk check
      runs regardless of what it says.
    </>
  ),

  arbitrage: (
    <>
      <strong>Prices that contradict each other.</strong> If &ldquo;yes&rdquo; costs 40¢
      and &ldquo;no&rdquo; costs 45¢ on the same question, buying both costs 85¢ and is
      guaranteed to return exactly $1 — because one of them must happen.
      <br />
      <br />
      Genuine cases are rare and small, and fees eat most of them. Everything here is
      priced <em>after</em> fees and after checking there are enough contracts actually
      available on every leg — because an arbitrage where one leg fails to fill is not an
      arbitrage, it is an accidental bet.
    </>
  ),

  news: (
    <>
      <strong>News the platform has read, and which markets it might affect.</strong> It
      polls free public sources and tries to match stories to markets.
      <br />
      <br />
      Two timestamps are shown and the difference matters. <em>Published</em> is when the
      source says it went out; <em>ingested</em> is when this platform actually received
      it. Only the second is used for decisions — you cannot trade on news you had not
      received yet, and pretending otherwise makes historical testing meaningless.
      <br />
      <br />
      &ldquo;Matched to no market&rdquo; is the normal, common result.
    </>
  ),

  risk: (
    <>
      <strong>The limits that stop things going badly wrong.</strong> Every proposed trade
      is checked against all of them and refused if it breaks any.
      <br />
      <br />
      Refusals are not failures — they are the system working. A platform that never
      refuses its own trades has no limits.
      <br />
      <br />
      The 1× leverage ceiling on perpetuals is different from everything else here: it is
      not a setting with a cautious value, it is a rule with no number to change.
    </>
  ),

  controls: (
    <>
      <strong>The stop buttons.</strong>
      <br />
      <br />
      A <em>kill switch</em> halts trading and stays halted — including through a restart,
      so it cannot be undone by turning something off and on again. There are separate
      switches per domain, so you can stop one kind of trading without stopping the other.
      <br />
      <br />
      <em>Operating mode</em> decides how much the platform may do without asking. Nothing
      here can bypass a risk limit or the leverage ceiling; these controls only ever make
      it do <em>less</em>.
    </>
  ),

  system: (
    <>
      <strong>Whether the machinery itself is healthy.</strong> Not about profit — about
      whether the parts are running and keeping up.
      <br />
      <br />
      Each component reports its own health. If a critical one fails, trading stops
      automatically and stays stopped until the cause is cleared. Data that arrives too
      late to be reliable is discarded rather than traded on.
    </>
  ),

  audit: (
    <>
      <strong>A permanent record of every decision.</strong> Including the trades it
      considered and declined, and why.
      <br />
      <br />
      This is the answer to &ldquo;why did it do that?&rdquo; — and equally to &ldquo;why
      did it not?&rdquo;. Every human action is recorded here too, with who did it and
      when.
    </>
  ),
};
