"use client";

/**
 * Requirement 29: every material trading decision must be inspectable.
 *
 * Decisions sit alongside the control-action log because "why did it not trade?"
 * is asked at least as often as "why did it trade?", and the declined decisions
 * are where that answer lives.
 */

import { useState } from "react";
import { api, timeAgo, type AuditRow, type Decision } from "@/lib/api";
import { DataTable, ErrorBox, Help, Pill, usePoll } from "@/components/ui";
import { PAGE } from "@/lib/explainers";

export default function AuditPage() {
  const [tab, setTab] = useState<"decisions" | "audit">("decisions");
  const { data: decisions, error } = usePoll<{ decisions: Decision[] }>(
    () => api.decisions(),
    3000,
  );
  const { data: audit } = usePoll<{ entries: AuditRow[] }>(() => api.audit(72), 5000);

  return (
    <>
      <h1>
        Audit log <Help wide>{PAGE.audit}</Help>
      </h1>
      <div className="sub">Every considered trade and every control action, with reasoning.</div>
      <ErrorBox error={error} />

      <div className="row" style={{ marginBottom: 12 }}>
        <button onClick={() => setTab("decisions")} className={tab === "decisions" ? "go" : ""}>
          Decisions
        </button>
        <button onClick={() => setTab("audit")} className={tab === "audit" ? "go" : ""}>
          Control actions
        </button>
      </div>

      {tab === "decisions" ? (
        <DataTable
          rows={decisions?.decisions ?? []}
          rowKey={(d) => d.decision_id}
          empty="No decisions recorded yet."
          columns={[
            { header: "When", cell: (d) => timeAgo(d.ts) },
            { header: "Market", cell: (d) => <span className="mono">{d.ticker}</span> },
            { header: "Domain", cell: (d) => <Pill>{d.market_type}</Pill> },
            { header: "Strategy", cell: (d) => d.strategy_id ?? "--" },
            {
              header: "Decision",
              cell: (d) => (
                <Pill
                  tone={
                    d.decision === "rejected" ? "bad" : d.decision === "resized" ? "warn" : "good"
                  }
                >
                  {d.decision}
                </Pill>
              ),
            },
            {
              header: "Reason",
              cell: (d) => <span className="mono small">{d.reason_code ?? ""}</span>,
            },
            { header: "Price", num: true, cell: (d) => d.market_price ?? "--" },
            { header: "Fair value", num: true, cell: (d) => d.fair_value ?? "--" },
            {
              header: "Edge",
              num: true,
              cell: (d) => (d.edge_cc === null ? "--" : `${d.edge_cc}cc`),
            },
            { header: "EV", num: true, cell: (d) => d.expected_value_usd ?? "--" },
            { header: "Req", num: true, cell: (d) => d.requested },
            { header: "Appr", num: true, cell: (d) => d.approved },
            {
              header: "Leverage",
              num: true,
              cell: (d) => (d.projected_leverage ? `${d.projected_leverage}x` : "--"),
            },
            {
              header: "Order",
              cell: (d) => <span className="mono small">{d.order_id ?? "--"}</span>,
            },
          ]}
        />
      ) : (
        <DataTable
          rows={audit?.entries ?? []}
          rowKey={(_a, i) => String(i)}
          empty="No control actions recorded."
          columns={[
            { header: "When", cell: (a) => timeAgo(a.ts) },
            { header: "Actor", cell: (a) => a.actor },
            { header: "Action", cell: (a) => <span className="mono">{a.action}</span> },
            { header: "Subject", cell: (a) => a.subject ?? "--" },
            {
              header: "Severity",
              cell: (a) => (
                <Pill
                  tone={
                    a.severity === "critical"
                      ? "bad"
                      : a.severity === "warning"
                        ? "warn"
                        : "neutral"
                  }
                >
                  {a.severity}
                </Pill>
              ),
            },
            {
              header: "Detail",
              cell: (a) => (
                <span className="small mono">
                  {a.detail ? JSON.stringify(a.detail).slice(0, 160) : ""}
                </span>
              ),
            },
          ]}
        />
      )}
    </>
  );
}
