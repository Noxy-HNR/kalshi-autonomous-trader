"use client";

/**
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
