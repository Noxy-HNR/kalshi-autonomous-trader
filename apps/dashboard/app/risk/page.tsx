"use client";

/**
 * Every risk limit, its utilisation, and everything that fired.
 */

import { api, timeAgo, type DocsModel, type RiskEventRow } from "@/lib/api";
import { DataTable, ErrorBox, Help, Pill, docsLink, usePoll } from "@/components/ui";
import { PAGE } from "@/lib/explainers";

type Limits = Record<string, unknown>;

/**
 * The reference entry for one row of the limits table.
 *
 * A field is either a slider (documented on the tunable) or configuration-only
 * (documented in risk_fields). Rows with neither get no tooltip rather than an
 * invented one.
 */
function docFor(
  docs: DocsModel | null,
  domain: string,
  field: string,
): { anchor: string; text: string } | null {
  if (!docs) return null;
  const tunable = docs.tunables.find((t) => t.key === `${domain}.${field}`);
  if (tunable) return { anchor: tunable.anchor, text: tunable.plain || tunable.description };

  const other = docs.risk_fields.find((f) => f.field === field);
  if (other) return { anchor: other.anchor, text: other.body ?? other.short ?? "" };

  // Several perpetual limits are the exact counterparts of event ones -- same
  // meaning, different domain -- and only the event side is adjustable from
  // the dashboard. Reuse that explanation rather than leaving the row bare or
  // duplicating the prose, and say which domain it is describing.
  const twin = docs.tunables.find((t) => t.key === `event.${field}`);
  if (twin) {
    return {
      anchor: twin.anchor,
      text: `${twin.plain || twin.description}

This row is the perpetuals equivalent of the same limit, set in configuration rather than from the dashboard.`,
    };
  }
  return null;
}

export default function RiskPage() {
  const { data: limits, error } = usePoll<Limits>(() => api.riskLimits(), 4000);
  const { data: events } = usePoll<{ events: RiskEventRow[] }>(() => api.riskEvents(), 4000);
  const lev = limits?.perp_leverage as Record<string, string> | undefined;
  const { data: docs } = usePoll<DocsModel>(() => api.docs(), 60000);

  return (
    <>
      <h1>
        Risk <Help wide>{PAGE.risk}</Help>
      </h1>
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
                    {Object.entries(block).map(([k, v]) => {
                      // Every row gets an explanation and a link into the
                      // reference. Which section it lives in depends on whether
                      // it is adjustable from the dashboard.
                      const doc = docFor(docs, domain, k);
                      return (
                        <tr key={k} {...(doc ? docsLink(doc.anchor) : {})}>
                          <td className="muted">
                            {k.replace(/_/g, " ")}
                            {doc?.text ? <Help wide>{doc.text}</Help> : null}
                          </td>
                          <td className="num">{String(v)}</td>
                        </tr>
                      );
                    })}
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
