"use client";

/**
 * Live market inventory.
 *
 * Freshness is a column because a market that is not fresh is not traded: stale
 * data produces no decision at all rather than a cautious one.
 */

import { api, contracts, type MarketRow } from "@/lib/api";
import { DataTable, ErrorBox, Help, Pill, usePoll } from "@/components/ui";
import { PAGE } from "@/lib/explainers";

export default function MarketsPage() {
  const { data, error } = usePoll<{ markets: MarketRow[]; count: number }>(
    () => api.markets(),
    2500,
  );
  const markets = data?.markets ?? [];
  const stale = markets.filter((m) => !m.fresh).length;

  return (
    <>
      <h1>
        Markets <Help wide>{PAGE.markets}</Help>
      </h1>
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
