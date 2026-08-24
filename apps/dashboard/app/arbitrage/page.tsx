"use client";

/**
 * Arbitrage opportunities, including the ones that were declined.
 *
 * The declined rows are the point. If the platform never declines anything its
 * cost model is understating costs; the distribution of declined edges is the
 * only direct evidence the fee, depth and leg-risk assumptions are calibrated.
 */

import { api, timeAgo, type ArbRow } from "@/lib/api";
import { DataTable, ErrorBox, Pill, usePoll } from "@/components/ui";

export default function ArbitragePage() {
  const { data, error } = usePoll<{ opportunities: ArbRow[] }>(() => api.arbitrage(), 4000);
  const rows = data?.opportunities ?? [];
  const actioned = rows.filter((r) => r.actioned).length;

  return (
    <>
      <h1>Arbitrage</h1>
      <div className="sub">
        Logical pricing inconsistencies, priced after fees, depth and leg risk.
      </div>
      <ErrorBox error={error} />

      <div className="row" style={{ marginBottom: 12 }}>
        <Pill tone="good">{actioned} actionable</Pill>
        <Pill>{rows.length - actioned} declined</Pill>
      </div>

      <DataTable
        rows={rows}
        rowKey={(r) => r.opportunity_id}
        empty="No opportunities evaluated yet."
        columns={[
          { header: "When", cell: (r) => timeAgo(r.ts) },
          { header: "Kind", cell: (r) => <span className="mono small">{r.kind}</span> },
          { header: "Domain", cell: (r) => <Pill>{r.market_type}</Pill> },
          { header: "Legs", num: true, cell: (r) => (r.legs?.length ?? 0) },
          { header: "Gross", num: true, cell: (r) => r.gross_edge_usd ?? "--" },
          { header: "Fees", num: true, cell: (r) => r.fees_usd ?? "--" },
          { header: "Net", num: true, cell: (r) => r.net_edge_usd ?? "--" },
          {
            header: "p(fill)",
            num: true,
            cell: (r) =>
              r.execution_probability === null ? "--" : r.execution_probability.toFixed(3),
          },
          { header: "Worst case", num: true, cell: (r) => r.worst_case_usd ?? "--" },
          { header: "EV", num: true, cell: (r) => r.expected_value_usd ?? "--" },
          {
            header: "Outcome",
            cell: (r) => (
              <Pill tone={r.actioned ? "good" : "neutral"}>
                {r.actioned ? "actionable" : "declined"}
              </Pill>
            ),
          },
          { header: "Why not", cell: (r) => <span className="small">{r.reason ?? ""}</span> },
        ]}
      />
    </>
  );
}
