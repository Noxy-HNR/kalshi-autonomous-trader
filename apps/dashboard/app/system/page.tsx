"use client";

/**
 * Process, host, queues, latency and data freshness.
 */

import { api, type SystemStatus } from "@/lib/api";
import { ErrorBox, Help, Pill, Stat, docsLink, usePoll } from "@/components/ui";
import { PAGE } from "@/lib/explainers";

function histogram(
  metrics: SystemStatus["metrics"] | undefined,
  prefix: string,
): Array<[string, Record<string, number>]> {
  if (!metrics) return [];
  return Object.entries(metrics.histograms).filter(([k]) => k.startsWith(prefix));
}

export default function SystemPage() {
  const { data, error } = usePoll<SystemStatus>(() => api.system(), 3000);
  const m = data?.metrics;

  return (
    <>
      <h1>
        System <Help wide>{PAGE.system}</Help>
      </h1>
      <div className="sub">Resources, queues, latency and feed health.</div>
      <ErrorBox error={error} />

      {data ? (
        <>
          <div className="grid g4">
            <Stat
              label="Uptime"
              value={`${Math.floor((m?.uptime_s ?? 0) / 60)}m`}
              small
            />
            <Stat label="Process RSS" value={`${data.process.rss_mb ?? 0} MB`} small />
            <Stat
              label="Host CPU"
              value={`${data.host.cpu_pct ?? 0}%`}
              tone={(data.host.cpu_pct ?? 0) > 85 ? "warn" : undefined}
              small
            />
            <Stat
              label="Host RAM"
              value={`${data.host.ram_pct ?? 0}%`}
              tone={(data.host.ram_pct ?? 0) > 90 ? "warn" : undefined}
              small
            />
            <Stat
              label="Disk free"
              value={`${data.host.disk_free_gb ?? 0} GB`}
              tone={(data.host.disk_free_gb ?? 0) < 5 ? "bad" : undefined}
              small
            />
            <Stat label="Threads" value={String(data.process.threads ?? 0)} small />
          </div>

          <h2 {...docsLink("control-remote-access")}>
            Remote access <Help wide>{PAGE.remoteAccess}</Help>
          </h2>
          <div className="card">
            {(() => {
              const ra = data.remote_access;
              if (!ra) return <div className="muted small">Not reported by this engine.</div>;
              if (ra.connected)
                return (
                  <>
                    <div className="row" style={{ gap: 10, alignItems: "center" }}>
                      <Pill tone="good">reachable</Pill>
                      <code className="mono">{ra.address}</code>
                    </div>
                    <div className="small muted" style={{ marginTop: 8 }}>
                      Open the dashboard from any device on your tailnet at{" "}
                      <code className="mono">http://{ra.address}:3030</code>. Only devices you
                      have authorised can reach it; there is no route from the public internet.
                    </div>
                  </>
                );
              if (ra.env_override !== "tailscale")
                return (
                  <>
                    <Pill tone="neutral">off &mdash; this machine only</Pill>
                    <div className="small muted" style={{ marginTop: 8 }}>
                      Set <code className="mono">KAT_API_REMOTE_ACCESS=tailscale</code> in{" "}
                      <code className="mono">.env</code> and restart to reach this from your
                      phone or another laptop.
                    </div>
                  </>
                );
              return (
                <>
                  <Pill tone="warn">requested, not connected</Pill>
                  <div className="small muted" style={{ marginTop: 8 }}>
                    {ra.installed
                      ? "Tailscale is installed but this machine has no tailnet address — run `tailscale up`, then restart the platform."
                      : "Tailscale is not installed on this machine. Install it, run `tailscale up`, then restart the platform."}{" "}
                    The dashboard stays on loopback until then, which is the safe direction.
                  </div>
                </>
              );
            })()}
          </div>

          <h2>Market data feeds</h2>
          <div className="grid g2">
            {Object.entries(data.market_data ?? {}).map(([domain, stats]) => (
              <div key={domain} className="card">
                <h3>{domain}</h3>
                <table>
                  <tbody>
                    {Object.entries(stats).map(([k, v]) => (
                      <tr key={k}>
                        <td className="muted">{k.replace(/_/g, " ")}</td>
                        <td className="num">{v === null ? "--" : String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>

          <h2>Latency</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="num">count</th>
                  <th className="num">p50</th>
                  <th className="num">p95</th>
                  <th className="num">p99</th>
                  <th className="num">max</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ...histogram(m, "kalshi.rest.latency_ms"),
                  ...histogram(m, "execution.submit_latency_ms"),
                  ...histogram(m, "risk.evaluation_latency_ms"),
                  ...histogram(m, "engine.decision_latency_ms"),
                  ...histogram(m, "model.inference_latency_ms"),
                ].map(([name, h]) => (
                  <tr key={name}>
                    <td className="mono small">{name}</td>
                    <td className="num">{h.count}</td>
                    <td className="num">{h.p50}</td>
                    <td className="num">{h.p95}</td>
                    <td className="num">{h.p99}</td>
                    <td className="num">{h.max}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="note">
            Tail percentiles, not means. On an execution path the number that matters is the
            tail, and a mean hides it.
          </div>

          <h2>Counters</h2>
          <div className="card">
            <div className="row small">
              {Object.entries(m?.counters ?? {}).map(([k, v]) => (
                <Pill key={k} tone={k.includes("leverage_blocks") && v > 0 ? "warn" : "neutral"}>
                  {k}: {v}
                </Pill>
              ))}
            </div>
            {Object.keys(m?.counters ?? {}).length === 0 ? (
              <div className="muted small">No counters recorded yet.</div>
            ) : null}
          </div>
        </>
      ) : (
        !error && <div className="empty">Connecting to the trading engine...</div>
      )}
    </>
  );
}
