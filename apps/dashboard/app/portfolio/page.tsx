"use client";

/**
 * Current portfolio, visually.
 *
 * Every chart on this page is built from the same endpoints the rest of the
 * dashboard already polls (`/overview`, `/events/positions`, `/perps/positions`)
 * -- there is no separate "pretty" data source. The equity-over-time chart is
 * the one piece that doesn't exist server-side: the engine doesn't persist an
 * equity time series, so it's built client-side from real observations as they
 * arrive and kept in this browser's localStorage. That's stated on the page
 * rather than left to look like more history than it is.
 */

import { api, usd, type Overview } from "@/lib/api";
import { AreaChart, DonutChart, HBarChart, useSeries } from "@/components/charts";
import { ErrorBox, Help, Stat, usePoll } from "@/components/ui";
import { PAGE } from "@/lib/explainers";

interface AnyPosition {
  ticker: string;
  exposure_usd?: string;
  notional_usd?: string;
  unrealized_pnl_usd?: string;
}

export default function PortfolioPage() {
  const { data, error } = usePoll<Overview>(() => api.overview(), 2000);
  const { data: events } = usePoll<{ positions: AnyPosition[] }>(
    () => api.eventPositions() as Promise<{ positions: AnyPosition[] }>,
    3000,
  );
  const { data: perps } = usePoll<{ positions: AnyPosition[] }>(
    () => api.perpPositions() as Promise<{ positions: AnyPosition[] }>,
    3000,
  );

  const equitySeries = useSeries("kat.chart.total_equity", data, (d) =>
    Number(d.total_equity_usd),
  );
  const eventEquitySeries = useSeries("kat.chart.event_equity", data, (d) =>
    Number(d.event.equity_usd ?? "0"),
  );
  const perpEquitySeries = useSeries("kat.chart.perp_equity", data, (d) =>
    Number(d.perp.equity_usd ?? "0"),
  );

  const eventEquity = Number(data?.event.equity_usd ?? 0);
  const perpEquity = Number(data?.perp.equity_usd ?? 0);
  const eventExposure = Number(data?.event.exposure_usd ?? 0);
  const perpExposure = Number(data?.perp.exposure_usd ?? 0);
  const eventPnl = Number(data?.event.realized_pnl_today_usd ?? 0);
  const perpPnl = Number(data?.perp.realized_pnl_today_usd ?? 0);

  const positionBars = [
    ...(events?.positions ?? []).map((p) => ({
      label: p.ticker,
      value: Number(p.exposure_usd ?? 0),
      tone: (Number(p.unrealized_pnl_usd ?? 0) > 0
        ? "good"
        : Number(p.unrealized_pnl_usd ?? 0) < 0
          ? "bad"
          : "accent") as "good" | "bad" | "accent",
    })),
    ...(perps?.positions ?? []).map((p) => ({
      label: p.ticker,
      value: Number(p.notional_usd ?? 0),
      tone: "warn" as const,
    })),
  ]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  return (
    <>
      <h1>
        Portfolio <Help wide>{PAGE.portfolio}</Help>
      </h1>
      <div className="sub">Current equity, allocation and exposure across both domains.</div>
      <ErrorBox error={error} />

      {data ? (
        <>
          <div className="grid g4">
            <Stat label="Total equity" value={usd(data.total_equity_usd)} />
            <Stat
              label="Total exposure"
              value={usd(String(eventExposure + perpExposure))}
              note="event + perp, open and pending"
              small
            />
            <Stat
              label="Realised P&L today"
              value={usd(String(eventPnl + perpPnl))}
              tone={eventPnl + perpPnl > 0 ? "good" : eventPnl + perpPnl < 0 ? "bad" : undefined}
              small
            />
            <Stat
              label="Open positions"
              value={String((data.event.open_positions ?? 0) + (data.perp.open_positions ?? 0))}
              note={`${data.event.open_positions ?? 0} event · ${data.perp.open_positions ?? 0} perp`}
              small
            />
          </div>

          <h2>Equity over this session</h2>
          <div className="card">
            <AreaChart points={equitySeries} format={(v) => usd(String(v))} color="var(--accent)" />
            <div className="note" style={{ marginBottom: 0 }}>
              Recorded from live polls in this browser as the dashboard stays open — not a
              server-side history, so it starts over on a new device or after local data is
              cleared.
            </div>
          </div>

          <h2>Allocation</h2>
          <div className="grid g2">
            <div className="card">
              <h3>Equity by domain</h3>
              <DonutChart
                centerValue={usd(data.total_equity_usd, 0)}
                centerLabel="total equity"
                segments={[
                  { label: "Event contracts", value: Math.max(0, eventEquity), color: "var(--accent)" },
                  { label: "Perpetuals", value: Math.max(0, perpEquity), color: "var(--warn)" },
                ]}
              />
            </div>
            <div className="card">
              <h3>Exposure by domain</h3>
              <DonutChart
                centerValue={usd(String(eventExposure + perpExposure), 0)}
                centerLabel="total exposure"
                segments={[
                  { label: "Event contracts", value: Math.max(0, eventExposure), color: "var(--good)" },
                  { label: "Perpetuals", value: Math.max(0, perpExposure), color: "var(--bad)" },
                ]}
              />
            </div>
          </div>

          <h2>Top positions by size</h2>
          <div className="card">
            <HBarChart
              rows={positionBars}
              format={(v) => usd(String(v))}
              emptyLabel="No open positions in either domain."
            />
          </div>

          <h2>Today&rsquo;s P&amp;L by domain</h2>
          <div className="card">
            <HBarChart
              rows={[
                { label: "Event contracts", value: eventPnl, tone: eventPnl >= 0 ? "good" : "bad" },
                { label: "Perpetuals", value: perpPnl, tone: perpPnl >= 0 ? "good" : "bad" },
              ]}
              format={(v) => usd(String(v))}
            />
          </div>

          <h2>Equity trend by domain</h2>
          <div className="grid g2">
            <div className="card">
              <h3>Event equity</h3>
              <AreaChart
                points={eventEquitySeries}
                format={(v) => usd(String(v))}
                color="var(--accent)"
                height={64}
              />
            </div>
            <div className="card">
              <h3>Perp equity</h3>
              <AreaChart
                points={perpEquitySeries}
                format={(v) => usd(String(v))}
                color="var(--warn)"
                height={64}
              />
            </div>
          </div>
        </>
      ) : (
        !error && <div className="empty">Connecting to the trading engine…</div>
      )}
    </>
  );
}
