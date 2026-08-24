"use client";

/**
 * Every risk limit, its utilisation, and everything that fired.
 */

import { api, timeAgo, type RiskEventRow } from "@/lib/api";
import { DataTable, ErrorBox, Pill, usePoll } from "@/components/ui";

type Limits = Record<string, unknown>;

export default function RiskPage() {
  const { data: limits, error } = usePoll<Limits>(() => api.riskLimits(), 4000);
  const { data: events } = usePoll<{ events: RiskEventRow[] }>(() => api.riskEvents(), 4000);
  const lev = limits?.perp_leverage as Record<string, string> | undefined;

  return (
    <>
      <h1>Risk</h1>
      <div className="sub">Limits, current utilisation, remaining capacity and triggers.</div>
      <ErrorBox error={error} />

      <div className="card" style={{ borderColor: "rgba(62,207,142,.35)" }}>
        <h3>Perpetual leverage</h3>
        <div className="row" style={{ gap: 18, alignItems: "baseline" }}>
          <span className="stat good">{lev?.limit ?? "1.00x"}</span>
          <span className="muted small">
            {lev?.note ?? "architectural invariant; not a setting and cannot be changed"}
          </span>
        </div>
        {lev ? (
          <div className="row small muted" style={{ marginTop: 8 }}>
            <span>current {lev.current_leverage ?? "--"}x</span>
            <span>projected {lev.projected_leverage ?? "--"}x</span>
            <span>equity {lev.equity_usd ?? "--"}</span>
            <span>headroom {lev.headroom_usd ?? "--"}</span>
          </div>
        ) : null}
      </div>

      <h2>Configured limits</h2>
      <div className="grid g2">
        {(["event", "perp"] as const).map((domain) => {
          const block = limits?.[domain] as Record<string, unknown> | undefined;
          const use = limits?.[`${domain}_utilisation`] as Record<string, string> | undefined;
          return (
            <div key={domain} className="card">
              <h3>{domain === "event" ? "Event contracts" : "Perpetuals"}</h3>
              {use ? (
                <div className="row small" style={{ marginBottom: 10 }}>
                  {Object.entries(use).map(([k, v]) => (
                    <Pill key={k}>
                      {k.replace(/_/g, " ")}: {v}
                    </Pill>
                  ))}
                </div>
              ) : null}
              {block ? (
                <table>
                  <tbody>
                    {Object.entries(block).map(([k, v]) => (
                      <tr key={k}>
                        <td className="muted">{k.replace(/_/g, " ")}</td>
                        <td className="num">{String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="muted small">Not available.</div>
              )}
              <div className="note">
                Clamped by the backend on load. These fields only let you be more conservative
                than the platform default; they cannot loosen a limit.
              </div>
            </div>
          );
        })}
      </div>

      <h2>Risk events</h2>
      <DataTable
        rows={events?.events ?? []}
        rowKey={(_e, i) => String(i)}
        empty="No risk events recorded."
        columns={[
          { header: "When", cell: (e) => timeAgo(e.ts) },
          {
            header: "Severity",
            cell: (e) => (
              <Pill
                tone={
                  e.severity === "critical" ? "bad" : e.severity === "warning" ? "warn" : "neutral"
                }
              >
                {e.severity}
              </Pill>
            ),
          },
          { header: "Domain", cell: (e) => e.market_type },
          { header: "Market", cell: (e) => <span className="mono">{e.ticker ?? "--"}</span> },
          { header: "Reason", cell: (e) => <span className="mono small">{e.reason_code}</span> },
          { header: "Limit", num: true, cell: (e) => e.limit_value ?? "--" },
          { header: "Observed", num: true, cell: (e) => e.observed_value ?? "--" },
          { header: "Message", cell: (e) => <span className="small">{e.message ?? ""}</span> },
        ]}
      />
    </>
  );
}
