"use client";

import { api, usd, type Overview } from "@/lib/api";
import { EnvironmentBanner, ErrorBox, Pill, Stat, usePoll } from "@/components/ui";

export default function OverviewPage() {
  const { data, error } = usePoll<Overview>(() => api.overview(), 2000);

  return (
    <>
      <h1>Overview</h1>
      <div className="sub">Account, exposure and health across both trading domains.</div>
      <ErrorBox error={error} />

      {data ? (
        <>
          <EnvironmentBanner
            environment={data.environment}
            canSubmitRealOrders={data.can_submit_real_orders}
            tradingPermitted={data.trading_permitted}
            phase={data.phase}
            mode={data.operating_mode}
          />

          <div className="grid g4">
            <Stat label="Total equity" value={usd(data.total_equity_usd)} />
            <Stat
              label="Event equity"
              value={usd(data.event.equity_usd)}
              note={`${data.event.open_positions ?? 0} positions · ${data.event.open_orders ?? 0} orders`}
              small
            />
            <Stat
              label="Perp equity"
              value={usd(data.perp.equity_usd)}
              note={
                data.perp.equity_is_known === false
                  ? "equity unknown — perp trading blocked"
                  : `source: ${data.perp.equity_source ?? "unknown"}`
              }
              tone={data.perp.equity_is_known === false ? "bad" : undefined}
              small
            />
            <Stat
              label="Effective perp leverage"
              value={
                data.perp.effective_leverage
                  ? `${Number(data.perp.effective_leverage).toFixed(4)}×`
                  : "0.0000×"
              }
              note={`hard maximum ${data.perp.max_leverage ?? "1.00"}×`}
              tone={
                data.perp.effective_leverage && Number(data.perp.effective_leverage) > 1
                  ? "bad"
                  : "good"
              }
              small
            />
          </div>

          <h2>Today</h2>
          <div className="grid g4">
            <Stat
              label="Event realised P&L"
              value={usd(data.event.realized_pnl_today_usd)}
              tone={pnlTone(data.event.realized_pnl_today_usd)}
              small
            />
            <Stat
              label="Perp realised P&L"
              value={usd(data.perp.realized_pnl_today_usd)}
              tone={pnlTone(data.perp.realized_pnl_today_usd)}
              small
            />
            <Stat
              label="Event exposure"
              value={usd(data.event.exposure_usd)}
              note={`drawdown ${data.event.drawdown_pct ?? "0"}`}
              small
            />
            <Stat
              label="Perp exposure"
              value={usd(data.perp.exposure_usd)}
              note={`pending ${usd(data.perp.pending_exposure_usd)}`}
              small
            />
          </div>

          <h2>Activity</h2>
          <div className="grid g2">
            {(["event", "perp"] as const).map((domain) => {
              const summary = data[domain];
              const cycle = summary.last_cycle;
              return (
                <div key={domain} className="card">
                  <h3>{domain === "event" ? "Event contracts" : "Perpetuals"}</h3>
                  {!summary.available ? (
                    <div className="muted small">Domain is not active.</div>
                  ) : (
                    <>
                      <div className="row small" style={{ marginBottom: 10 }}>
                        <Pill tone={summary.strategies_active ? "good" : "neutral"}>
                          {summary.strategies_active ?? 0} active /{" "}
                          {summary.strategies_registered ?? 0} strategies
                        </Pill>
                        <Pill>{summary.markets_watched ?? 0} markets watched</Pill>
                        <Pill>{summary.cycles ?? 0} cycles</Pill>
                      </div>
                      {cycle ? (
                        <table>
                          <tbody>
                            <Row label="Markets examined" value={cycle.markets_examined} />
                            <Row
                              label="Skipped as stale"
                              value={cycle.markets_skipped_stale}
                              warn={cycle.markets_skipped_stale > 0}
                            />
                            <Row label="Intents generated" value={cycle.intents_generated} />
                            <Row label="Approved" value={cycle.intents_approved} />
                            <Row label="Rejected" value={cycle.intents_rejected} />
                            <Row label="Orders submitted" value={cycle.orders_submitted} />
                            <Row
                              label="Errors"
                              value={cycle.errors}
                              warn={cycle.errors > 0}
                            />
                            <Row label="Cycle time" value={`${cycle.duration_ms.toFixed(2)} ms`} />
                          </tbody>
                        </table>
                      ) : (
                        <div className="muted small">No cycle has run yet.</div>
                      )}
                      {cycle && Object.keys(cycle.rejection_reasons).length > 0 ? (
                        <div className="note">
                          Last cycle declined:{" "}
                          {Object.entries(cycle.rejection_reasons)
                            .map(([reason, count]) => `${reason} (${count})`)
                            .join(", ")}
                          . Declining is the normal outcome — the platform is designed to be
                          comfortable taking no trade.
                        </div>
                      ) : null}
                    </>
                  )}
                </div>
              );
            })}
          </div>

          <h2>Health</h2>
          <div className="card">
            <div className="row" style={{ marginBottom: 10 }}>
              <Pill tone={data.health.database ? "good" : "bad"}>
                database {data.health.database ? "connected" : "unavailable"}
              </Pill>
              {(data.health.components ?? []).map((c) => (
                <Pill key={c.name} tone={c.healthy ? "good" : "bad"}>
                  {c.name}
                </Pill>
              ))}
            </div>
            {(data.health.kill_switches ?? []).filter((k) => k.engaged).length > 0 ? (
              <div className="err" style={{ marginBottom: 0 }}>
                <strong>Kill switches engaged:</strong>{" "}
                {(data.health.kill_switches ?? [])
                  .filter((k) => k.engaged)
                  .map((k) => `${k.scope}${k.subject ? `:${k.subject}` : ""} (${k.reason ?? "no reason"})`)
                  .join("; ")}
              </div>
            ) : (
              <div className="muted small">No kill switches engaged.</div>
            )}
            {data.health.writer ? (
              <div className="note">
                Persistence queue: {data.health.writer.queued}/{data.health.writer.capacity} queued,{" "}
                {data.health.writer.written} written, {data.health.writer.dropped} dropped. The
                database is never on the order path; writes drain in the background.
              </div>
            ) : null}
          </div>
        </>
      ) : (
        !error && <div className="empty">Connecting to the trading engine…</div>
      )}
    </>
  );
}

function Row({
  label,
  value,
  warn,
}: {
  label: string;
  value: string | number;
  warn?: boolean;
}) {
  return (
    <tr>
      <td className="muted">{label}</td>
      <td className="num" style={warn ? { color: "var(--warn)" } : undefined}>
        {value}
      </td>
    </tr>
  );
}

function pnlTone(value: string | undefined): "good" | "bad" | undefined {
  if (value === undefined) return undefined;
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return undefined;
  return n > 0 ? "good" : "bad";
}
