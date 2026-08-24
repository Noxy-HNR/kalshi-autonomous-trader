"""One-off generator for the remaining dashboard pages.

Kept in the repository because it documents exactly what was generated rather
than hand-written, and re-running it reproduces those files byte for byte.
"""

from __future__ import annotations

import pathlib

HEADER = '"use client";\n\n'

PAGES: dict[str, str] = {}

PAGES["markets"] = '''/**
 * Live market inventory.
 *
 * Freshness is a column because a market that is not fresh is not traded: stale
 * data produces no decision at all rather than a cautious one.
 */

import { api, contracts, type MarketRow } from "@/lib/api";
import { DataTable, ErrorBox, Pill, usePoll } from "@/components/ui";

export default function MarketsPage() {
  const { data, error } = usePoll<{ markets: MarketRow[]; count: number }>(
    () => api.markets(),
    2500,
  );
  const markets = data?.markets ?? [];
  const stale = markets.filter((m) => !m.fresh).length;

  return (
    <>
      <h1>Markets</h1>
      <div className="sub">Spreads, depth, volume and data freshness across both domains.</div>
      <ErrorBox error={error} />

      <div className="row" style={{ marginBottom: 12 }}>
        <Pill>{markets.length} watched</Pill>
        <Pill tone={stale ? "warn" : "good"}>{stale} stale</Pill>
      </div>

      <DataTable
        rows={markets}
        rowKey={(m) => m.ticker}
        empty="No markets are being watched. Market data needs Kalshi credentials -- the WebSocket is authenticated even for public channels."
        columns={[
          { header: "Market", cell: (m) => <span className="mono">{m.ticker}</span> },
          { header: "Domain", cell: (m) => <Pill>{m.market_type}</Pill> },
          { header: "Status", cell: (m) => m.status ?? "--" },
          { header: "Bid", num: true, cell: (m) => m.bid ?? "--" },
          { header: "Ask", num: true, cell: (m) => m.ask ?? "--" },
          { header: "Mid", num: true, cell: (m) => m.mid ?? "--" },
          { header: "Mark", num: true, cell: (m) => m.mark ?? "--" },
          {
            header: "Spread",
            num: true,
            cell: (m) => (m.spread_cc === null ? "--" : `${m.spread_cc}cc`),
          },
          { header: "Bid depth", num: true, cell: (m) => contracts(m.depth_bid_qc) },
          { header: "Ask depth", num: true, cell: (m) => contracts(m.depth_ask_qc) },
          { header: "Volume", num: true, cell: (m) => contracts(m.volume_qc) },
          {
            header: "Age",
            num: true,
            cell: (m) => (
              <span style={{ color: m.fresh ? undefined : "var(--warn)" }}>
                {m.age_ms === null ? "never" : `${m.age_ms}ms`}
              </span>
            ),
          },
        ]}
      />
    </>
  );
}
'''

PAGES["risk"] = '''/**
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
'''

PAGES["audit"] = '''/**
 * Requirement 29: every material trading decision must be inspectable.
 *
 * Decisions sit alongside the control-action log because "why did it not trade?"
 * is asked at least as often as "why did it trade?", and the declined decisions
 * are where that answer lives.
 */

import { useState } from "react";
import { api, timeAgo, type AuditRow, type Decision } from "@/lib/api";
import { DataTable, ErrorBox, Pill, usePoll } from "@/components/ui";

export default function AuditPage() {
  const [tab, setTab] = useState<"decisions" | "audit">("decisions");
  const { data: decisions, error } = usePoll<{ decisions: Decision[] }>(
    () => api.decisions(),
    3000,
  );
  const { data: audit } = usePoll<{ entries: AuditRow[] }>(() => api.audit(72), 5000);

  return (
    <>
      <h1>Audit log</h1>
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
'''

PAGES["system"] = '''/**
 * Process, host, queues, latency and data freshness.
 */

import { api, type SystemStatus } from "@/lib/api";
import { ErrorBox, Pill, Stat, usePoll } from "@/components/ui";

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
      <h1>System</h1>
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
'''

PAGES["models"] = '''/**
 * Model registry.
 *
 * Train, validation and test metrics are separate columns and are never blended.
 * Selection happens on the test column only; the train column is shown so the
 * overfitting gap between them is visible rather than hidden.
 */

import { api, type ModelRow } from "@/lib/api";
import { DataTable, ErrorBox, Pill, stateTone, usePoll } from "@/components/ui";

function num(block: Record<string, number> | null | undefined, key: string, digits = 4): string {
  const v = block?.[key];
  return v === null || v === undefined || !Number.isFinite(v) ? "--" : v.toFixed(digits);
}

export default function ModelsPage() {
  const { data, error } = usePoll<{ models: ModelRow[] }>(() => api.models(), 5000);
  const models = data?.models ?? [];

  return (
    <>
      <h1>Models</h1>
      <div className="sub">
        Versions, training windows, per-split metrics, calibration and deployment state.
      </div>
      <ErrorBox error={error} />

      <DataTable
        rows={models}
        rowKey={(m) => `${m.model_id}:${m.version}`}
        empty="No models registered yet. The trainer registers a model once it clears every out-of-sample gate."
        columns={[
          { header: "Model", cell: (m) => <span className="mono small">{m.model_id}</span> },
          { header: "Ver", cell: (m) => <span className="mono small">{m.version}</span> },
          { header: "Family", cell: (m) => m.model_type },
          { header: "Domain", cell: (m) => <Pill>{m.market_type}</Pill> },
          {
            header: "State",
            cell: (m) => <Pill tone={stateTone(m.deployment_state)}>{m.deployment_state}</Pill>,
          },
          { header: "Train AUC", num: true, cell: (m) => num(m.train_metrics, "auc") },
          { header: "Val AUC", num: true, cell: (m) => num(m.validation_metrics, "auc") },
          { header: "Test AUC", num: true, cell: (m) => num(m.test_metrics, "auc") },
          { header: "Test Brier", num: true, cell: (m) => num(m.test_metrics, "brier") },
          { header: "Test ECE", num: true, cell: (m) => num(m.test_metrics, "ece") },
          {
            header: "Excess ECE",
            num: true,
            cell: (m) => num(m.test_metrics, "excess_ece"),
          },
          {
            header: "Overfit gap",
            num: true,
            cell: (m) => {
              const train = m.train_metrics?.auc;
              const test = m.test_metrics?.auc;
              if (train === undefined || test === undefined) return "--";
              const gap = train - test;
              return (
                <span style={{ color: gap > 0.15 ? "var(--bad)" : undefined }}>
                  {gap.toFixed(4)}
                </span>
              );
            },
          },
          {
            header: "Calibrator",
            cell: (m) => String((m.calibration as Record<string, unknown>)?.method ?? "--"),
          },
          { header: "Commit", cell: (m) => <span className="mono small">{m.git_commit ?? "--"}</span> },
        ]}
      />

      <div className="note">
        Excess ECE is the calibration error left after subtracting what binomial sampling noise
        explains at that sample size. It is the number with economic meaning: a systematic
        two-point error manufactures a two-cent edge on a $0.50 market, whereas noise averages
        out across trades.
      </div>
    </>
  );
}
'''

PAGES["arbitrage"] = '''/**
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
'''

PAGES["news"] = '''/**
 * News and research.
 *
 * Both timestamps are shown. The gap between publication and ingestion is what a
 * backtest must respect: several free government feeds back-date items, so
 * filtering on the published time would hand a simulator hours of hindsight.
 */

import { api, timeAgo, type NewsRow } from "@/lib/api";
import { ErrorBox, Empty, Pill, usePoll } from "@/components/ui";

export default function NewsPage() {
  const { data, error } = usePoll<{ articles: NewsRow[] }>(() => api.news(), 10000);
  const articles = data?.articles ?? [];

  return (
    <>
      <h1>News &amp; research</h1>
      <div className="sub">
        Ingested articles, extracted entities, and the markets they were matched to.
      </div>
      <ErrorBox error={error} />

      {articles.length === 0 ? (
        <Empty>
          No articles ingested yet. The news pipeline polls free RSS and public government
          feeds; it never requires a paid API.
        </Empty>
      ) : (
        <div className="grid g2">
          {articles.map((a) => (
            <div key={a.article_id} className="card">
              <div className="row small muted" style={{ marginBottom: 6 }}>
                <Pill>{a.source}</Pill>
                <span>published {timeAgo(a.published_ts)}</span>
                <span>ingested {timeAgo(a.ingested_ts)}</span>
                {a.source_reliability !== null ? (
                  <span>reliability {a.source_reliability.toFixed(2)}</span>
                ) : null}
              </div>
              <div style={{ fontWeight: 600, marginBottom: 5 }}>
                {a.url ? (
                  <a href={a.url} target="_blank" rel="noreferrer">
                    {a.title}
                  </a>
                ) : (
                  a.title
                )}
              </div>
              {a.summary ? <div className="small muted">{a.summary.slice(0, 260)}</div> : null}
              {a.markets.length > 0 ? (
                <div className="row small" style={{ marginTop: 8 }}>
                  {a.markets.slice(0, 6).map((m) => (
                    <Pill key={m.ticker} tone={m.relevance > 0.6 ? "good" : "neutral"}>
                      {m.ticker} {m.relevance.toFixed(2)}
                    </Pill>
                  ))}
                </div>
              ) : (
                <div className="note">Matched to no market.</div>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  );
}
'''

PAGES["events"] = '''/**
 * Event contracts: positions, candidate markets and per-strategy attribution.
 */

import { api, usd, type DomainSummary, type Overview } from "@/lib/api";
import { DataTable, ErrorBox, Stat, usePoll } from "@/components/ui";

interface EventPosition {
  ticker: string;
  event_ticker: string;
  position: string;
  side: string;
  avg_entry: string | null;
  exposure_usd: string;
  unrealized_pnl_usd: string;
  strategy_id: string | null;
}

export default function EventsPage() {
  const { data: overview, error } = usePoll<Overview>(() => api.overview(), 3000);
  const { data: positions } = usePoll<{ available: boolean; positions: EventPosition[] }>(
    () => api.eventPositions() as Promise<{ available: boolean; positions: EventPosition[] }>,
    3000,
  );
  const summary: DomainSummary | undefined = overview?.event;

  return (
    <>
      <h1>Event contracts</h1>
      <div className="sub">
        Fully collateralised positions: maximum loss is known exactly at entry, so there is no
        leverage to constrain here -- only capital, concentration and loss budgets.
      </div>
      <ErrorBox error={error} />

      {summary ? (
        <div className="grid g4">
          <Stat label="Equity" value={usd(summary.equity_usd)} small />
          <Stat label="Cash" value={usd(summary.balance_usd)} small />
          <Stat label="Exposure" value={usd(summary.exposure_usd)} small />
          <Stat
            label="Realised today"
            value={usd(summary.realized_pnl_today_usd)}
            small
          />
        </div>
      ) : null}

      <h2>Open positions</h2>
      <DataTable
        rows={positions?.positions ?? []}
        rowKey={(p) => p.ticker}
        empty="No open event-contract positions."
        columns={[
          { header: "Market", cell: (p) => <span className="mono">{p.ticker}</span> },
          { header: "Event", cell: (p) => <span className="mono small">{p.event_ticker}</span> },
          { header: "Side", cell: (p) => p.side },
          { header: "Size", num: true, cell: (p) => p.position },
          { header: "Avg entry", num: true, cell: (p) => p.avg_entry ?? "--" },
          { header: "Exposure", num: true, cell: (p) => usd(p.exposure_usd) },
          {
            header: "Unrealised",
            num: true,
            cell: (p) => usd(p.unrealized_pnl_usd),
          },
          { header: "Strategy", cell: (p) => p.strategy_id ?? "--" },
        ]}
      />
    </>
  );
}
'''

PAGES["positions"] = '''/**
 * Positions across both domains, side by side.
 */

import { api, usd } from "@/lib/api";
import { DataTable, ErrorBox, Pill, usePoll } from "@/components/ui";

interface AnyPosition {
  ticker: string;
  position: string;
  exposure_usd?: string;
  notional_usd?: string;
  unrealized_pnl_usd?: string;
  avg_entry?: string | null;
  entry_price?: string | null;
  mark_price?: string | null;
  side?: string;
  is_long?: boolean;
}

export default function PositionsPage() {
  const { data: events, error } = usePoll<{ positions: AnyPosition[] }>(
    () => api.eventPositions() as Promise<{ positions: AnyPosition[] }>,
    3000,
  );
  const { data: perps } = usePoll<{ positions: AnyPosition[] }>(
    () => api.perpPositions() as Promise<{ positions: AnyPosition[] }>,
    3000,
  );

  return (
    <>
      <h1>Positions</h1>
      <div className="sub">Every open position in both domains.</div>
      <ErrorBox error={error} />

      <h2>Event contracts</h2>
      <DataTable
        rows={events?.positions ?? []}
        rowKey={(p) => `e-${p.ticker}`}
        empty="No open event-contract positions."
        columns={[
          { header: "Market", cell: (p) => <span className="mono">{p.ticker}</span> },
          { header: "Side", cell: (p) => <Pill>{p.side ?? "--"}</Pill> },
          { header: "Size", num: true, cell: (p) => p.position },
          { header: "Avg entry", num: true, cell: (p) => p.avg_entry ?? "--" },
          { header: "Exposure", num: true, cell: (p) => usd(p.exposure_usd) },
          { header: "Unrealised", num: true, cell: (p) => usd(p.unrealized_pnl_usd) },
        ]}
      />

      <h2>Perpetuals</h2>
      <DataTable
        rows={perps?.positions ?? []}
        rowKey={(p) => `p-${p.ticker}`}
        empty="No open perpetual positions."
        columns={[
          { header: "Market", cell: (p) => <span className="mono">{p.ticker}</span> },
          {
            header: "Side",
            cell: (p) => (
              <Pill tone={p.is_long ? "good" : "warn"}>{p.is_long ? "long" : "short"}</Pill>
            ),
          },
          { header: "Size", num: true, cell: (p) => p.position },
          { header: "Entry", num: true, cell: (p) => p.entry_price ?? "--" },
          { header: "Mark", num: true, cell: (p) => p.mark_price ?? "--" },
          { header: "Notional", num: true, cell: (p) => usd(p.notional_usd) },
        ]}
      />
    </>
  );
}
'''

PAGES["discovery"] = '''/**
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
'''


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1] / "apps" / "dashboard" / "app"
    for name, body in PAGES.items():
        path = root / name / "page.tsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER + body, encoding="utf-8")
        print(f"wrote {path.relative_to(root.parents[2])}")


if __name__ == "__main__":
    main()
