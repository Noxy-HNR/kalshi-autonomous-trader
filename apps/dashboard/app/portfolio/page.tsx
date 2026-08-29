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
import { ErrorBox, Help, Stat, docsLink, usePoll } from "@/components/ui";
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
            <Stat
              label="Total equity"
              value={usd(data.total_equity_usd)}
              docs="concept-equity"
              explain={
                "What the account is worth right now: cash plus what your open " +
                "positions would fetch if you closed them. Every limit on this " +
                "platform is a percentage of this number. Event contracts and " +
                "perpetuals hold money separately on Kalshi, so this is the two added together."
              }
            />
            <Stat
              label="Total exposure"
              value={usd(String(eventExposure + perpExposure))}
              note="event + perp, open and pending"
              small
              docs="concept-exposure"
              explain={
                "How much money is currently committed to positions rather than " +
                "sitting as cash. Zero means nothing can be lost to a price move. " +
                "It can never exceed your equity on perpetuals -- that is the 1x " +
                "leverage cap, the one limit with no setting."
              }
            />
            <Stat
              label="Realised P&L today"
              value={usd(String(eventPnl + perpPnl))}
              tone={eventPnl + perpPnl > 0 ? "good" : eventPnl + perpPnl < 0 ? "bad" : undefined}
              small
              docs="concept-pnl"
              explain={
                "Profit and loss from positions closed today. Realised means the " +
                "trade is finished and the money has actually moved -- unlike an " +
                "open position showing a gain, which is only worth that if you " +
                "close it now and pay the spread and fee to do so."
              }
            />
            <Stat
              label="Open positions"
              value={String((data.event.open_positions ?? 0) + (data.perp.open_positions ?? 0))}
              note={`${data.event.open_positions ?? 0} event · ${data.perp.open_positions ?? 0} perp`}
              small
              docs="concept-position"
              explain={
                "How many separate holdings are open. Each one ends either by you " +
                "selling it -- which needs a buyer, so thin markets are refused -- " +
                "or, for an event contract, by settling for $1.00 or nothing when " +
                "the question is answered."
              }
            />
          </div>

          <h2 {...docsLink("concept-equity")}>
            Equity over this session <Help wide>{"Your total equity as it changes. Recorded by this browser while the page is open, not stored on the server, so it starts over on another device."}</Help>
          </h2>
          <div className="card">
            <AreaChart points={equitySeries} format={(v) => usd(String(v))} color="var(--accent)" />
            <div className="note" style={{ marginBottom: 0 }}>
              Recorded from live polls in this browser as the dashboard stays open — not a
              server-side history, so it starts over on a new device or after local data is
              cleared.
            </div>
          </div>

          <h2 {...docsLink("concept-allocation")}>
            Allocation <Help wide>{"How the account is divided between the two domains. Splitting it is what stops one confidently wrong strategy from taking the whole account."}</Help>
          </h2>
          <div className="grid g2">
            <div className="card">
              <h3 {...docsLink("concept-equity")}>
                Equity by domain <Help>{"How your equity splits between the two domains right now."}</Help>
              </h3>
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
              <h3 {...docsLink("concept-exposure")}>
                Exposure by domain <Help>{"How much of each domain's money is committed to open positions."}</Help>
              </h3>
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

          <h2 {...docsLink("concept-position")}>
            Top positions by size <Help wide>{"The largest holdings you currently have open, biggest first. Size here is what is at stake, not what it has made or lost."}</Help>
          </h2>
          <div className="card">
            <HBarChart
              rows={positionBars}
              format={(v) => usd(String(v))}
              emptyLabel="No open positions in either domain."
            />
          </div>

          <h2 {...docsLink("concept-pnl")}>
            Today&rsquo;s P&amp;L by domain <Help wide>{"Profit and loss from positions closed today, split by domain. Only closed trades count here."}</Help>
          </h2>
          <div className="card">
            <HBarChart
              rows={[
                { label: "Event contracts", value: eventPnl, tone: eventPnl >= 0 ? "good" : "bad" },
                { label: "Perpetuals", value: perpPnl, tone: perpPnl >= 0 ? "good" : "bad" },
              ]}
              format={(v) => usd(String(v))}
            />
          </div>

          <h2 {...docsLink("concept-equity")}>
            Equity trend by domain <Help wide>{"Event and perpetual equity tracked separately, because on Kalshi they are genuinely separate pots of money that only you can move between."}</Help>
          </h2>
          <div className="grid g2">
            <div className="card">
              <h3 {...docsLink("concept-equity")}>
                Event equity <Help>{"Equity behind event contracts only."}</Help>
              </h3>
              <AreaChart
                points={eventEquitySeries}
                format={(v) => usd(String(v))}
                color="var(--accent)"
                height={64}
              />
            </div>
            <div className="card">
              <h3 {...docsLink("concept-equity")}>
                Perp equity <Help>{"Equity behind perpetuals only."}</Help>
              </h3>
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
