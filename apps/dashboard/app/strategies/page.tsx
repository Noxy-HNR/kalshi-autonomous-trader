"use client";

/**
 * Strategies and their lifecycle.
 *
 * The important design decision here is that performance is shown **per stage,
 * never merged**. A backtest Sharpe and a live Sharpe are different claims about
 * the world, and a table that averaged them would be actively misleading. Each
 * stage gets its own column and empty means "this stage has not run", not zero.
 */

import { api, usd, type StrategyRow } from "@/lib/api";
import { DataTable, ErrorBox, Pill, stateTone, usePoll } from "@/components/ui";

const LADDER = [
  "discovery",
  "development",
  "backtest",
  "out_of_sample",
  "walk_forward",
  "robustness",
  "paper",
  "probation",
  "live",
];

function metric(block: Record<string, unknown> | null | undefined, key: string): string {
  if (!block) return "--";
  const value = block[key];
  if (value === null || value === undefined) return "--";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : String(value);
}

export default function StrategiesPage() {
  const { data, error } = usePoll<{ strategies: StrategyRow[] }>(() => api.strategies(), 4000);
  const strategies = data?.strategies ?? [];

  return (
    <>
      <h1>Strategies</h1>
      <div className="sub">
        Lifecycle state and performance, reported separately for every stage.
      </div>
      <ErrorBox error={error} />

      <div className="card" style={{ marginBottom: 14 }}>
        <h3>Promotion ladder</h3>
        <div className="row small">
          {LADDER.map((state, i) => (
            <span key={state} className="row" style={{ gap: 6 }}>
              <Pill tone={stateTone(state)}>{state.replace(/_/g, " ")}</Pill>
              {i < LADDER.length - 1 ? <span className="muted">→</span> : null}
            </span>
          ))}
        </div>
        <div className="note">
          Promotion moves one rung at a time and only when that rung&apos;s gate passes.
          Demotion to paused or retired is always allowed. In gated discovery mode nothing
          reaches live capital without clearing every gate.
        </div>
      </div>

      <DataTable
        rows={strategies}
        rowKey={(s) => s.strategy_id}
        empty="No strategies registered yet."
        columns={[
          { header: "Strategy", cell: (s) => <span className="mono">{s.strategy_id}</span> },
          { header: "Domain", cell: (s) => <Pill>{s.market_type ?? "--"}</Pill> },
          { header: "Ver", cell: (s) => s.version ?? "--" },
          { header: "State", cell: (s) => <Pill tone={stateTone(s.state)}>{s.state}</Pill> },
          {
            header: "Enabled",
            cell: (s) => (
              <Pill tone={s.enabled ? "good" : "neutral"}>{s.enabled ? "yes" : "no"}</Pill>
            ),
          },
          { header: "Capital", num: true, cell: (s) => usd(s.allocated_capital_usd) },
          { header: "Backtest Sharpe", num: true, cell: (s) => metric(s.train_metrics, "sharpe") },
          { header: "OOS Sharpe", num: true, cell: (s) => metric(s.oos_metrics, "sharpe") },
          {
            header: "Walk-fwd Sharpe",
            num: true,
            cell: (s) => metric(s.walk_forward_metrics, "sharpe"),
          },
          { header: "Paper Sharpe", num: true, cell: (s) => metric(s.paper_metrics, "sharpe") },
          { header: "Live Sharpe", num: true, cell: (s) => metric(s.live_metrics, "sharpe") },
          { header: "Live trades", num: true, cell: (s) => metric(s.live_metrics, "trades") },
          {
            header: "Live drawdown",
            num: true,
            cell: (s) => metric(s.live_metrics, "max_drawdown_pct"),
          },
          {
            header: "Confidence",
            num: true,
            cell: (s) => (s.confidence === null || s.confidence === undefined ? "--" : s.confidence.toFixed(2)),
          },
          {
            header: "Retired because",
            cell: (s) => <span className="small">{s.retire_reason ?? ""}</span>,
          },
        ]}
      />

      <div className="note">
        A dash means that stage has not run, not a score of zero. Nothing on this page averages
        across stages: a backtest result and a live result are different claims and are kept
        that way.
      </div>
    </>
  );
}
