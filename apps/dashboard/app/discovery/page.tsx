"use client";

/**
 * Strategy discovery.
 *
 * Shows candidates and where each one sits on the promotion ladder. The
 * distinction that matters: a candidate with a good backtest has demonstrated
 * nothing yet, and this page is arranged so that is obvious at a glance.
 */

import { api, type StrategyRow } from "@/lib/api";
import { DataTable, ErrorBox, Pill, stateTone, usePoll } from "@/components/ui";

const PRE_LIVE = new Set([
  "discovery",
  "development",
  "backtest",
  "out_of_sample",
  "walk_forward",
  "robustness",
  "paper",
  "probation",
]);

export default function DiscoveryPage() {
  const { data, error } = usePoll<{ strategies: StrategyRow[] }>(() => api.strategies(), 5000);
  const candidates = (data?.strategies ?? []).filter((s) => PRE_LIVE.has(s.state));

  return (
    <>
      <h1>Strategy discovery</h1>
      <div className="sub">
        Candidates working their way up the ladder. Nothing here trades live capital.
      </div>
      <ErrorBox error={error} />

      <DataTable
        rows={candidates}
        rowKey={(s) => s.strategy_id}
        empty="No candidate strategies in the pipeline."
        columns={[
          { header: "Strategy", cell: (s) => <span className="mono">{s.strategy_id}</span> },
          { header: "Domain", cell: (s) => <Pill>{s.market_type ?? "--"}</Pill> },
          { header: "Stage", cell: (s) => <Pill tone={stateTone(s.state)}>{s.state}</Pill> },
          {
            header: "Backtest",
            cell: (s) => (s.train_metrics ? "run" : "--"),
          },
          {
            header: "Out-of-sample",
            cell: (s) => (s.oos_metrics ? "run" : "--"),
          },
          {
            header: "Walk-forward",
            cell: (s) => (s.walk_forward_metrics ? "run" : "--"),
          },
          {
            header: "Robustness",
            cell: (s) => (s.robustness_metrics ? "run" : "--"),
          },
          {
            header: "Paper",
            cell: (s) => (s.paper_metrics ? "run" : "--"),
          },
        ]}
      />

      <div className="note">
        A stage marked &quot;--&quot; has not run. Promotion requires the gate for the next rung
        to pass, one rung at a time, and a good backtest on its own is not evidence of anything.
      </div>
    </>
  );
}
