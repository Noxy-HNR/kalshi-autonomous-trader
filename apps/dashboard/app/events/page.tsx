"use client";

/**
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
