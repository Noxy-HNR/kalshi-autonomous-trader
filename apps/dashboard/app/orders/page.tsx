"use client";

/**
 * Orders as *our* state machine sees them, which is not the same as the
 * exchange's. UNKNOWN is surfaced prominently because an order in that state may
 * still be resting, still consumes exposure, and is never resubmitted -- only
 * reconciliation resolves it.
 */

import { useState } from "react";
import { api, timeAgo, usd, type OrderRow } from "@/lib/api";
import { DataTable, ErrorBox, Help, Pill, stateTone, usePoll } from "@/components/ui";
import { PAGE } from "@/lib/explainers";

const STATES = [
  "created",
  "submitted",
  "acknowledged",
  "partially_filled",
  "filled",
  "cancel_requested",
  "cancelled",
  "rejected",
  "risk_rejected",
  "expired",
  "unknown",
];

export default function OrdersPage() {
  const [filter, setFilter] = useState("");
  const { data, error } = usePoll<{ orders: OrderRow[] }>(() => api.orders(), 2500);
  const all = data?.orders ?? [];
  const orders = all.filter((o) => !filter || o.state === filter);
  const unknown = all.filter((o) => o.state === "unknown").length;

  return (
    <>
      <h1>
        Orders <Help wide>{PAGE.orders}</Help>
      </h1>
      <div className="sub">
        Submissions, acknowledgements, fills, rejections, slippage and latency.
      </div>
      <ErrorBox error={error} />

      {unknown > 0 ? (
        <div className="err">
          <strong>{unknown} order(s) in UNKNOWN state.</strong> The submission left this
          process but its outcome was never observed. They keep counting against exposure
          until reconciliation queries the exchange by client order id. They are never
          resubmitted &mdash; that is what would risk a doubled position.
        </div>
      ) : null}

      <div className="row" style={{ marginBottom: 12 }}>
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="">All states</option>
          {STATES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <span className="muted small">
          {orders.length} shown of {all.length}
        </span>
      </div>

      <DataTable
        rows={orders}
        rowKey={(o) => o.order_id}
        empty="No orders yet. In paper mode an order appears as soon as a strategy finds an edge that survives costs."
        columns={[
          { header: "When", cell: (o) => timeAgo(o.submitted_ts) },
          {
            header: "Market",
            cell: (o) => (
              <div>
                <div>{o.name ?? o.ticker}</div>
                <div className="mono small muted">{o.ticker}</div>
              </div>
            ),
          },
          { header: "Domain", cell: (o) => <Pill>{o.market_type}</Pill> },
          { header: "Side", cell: (o) => o.side ?? "--" },
          { header: "State", cell: (o) => <Pill tone={stateTone(o.state)}>{o.state}</Pill> },
          { header: "Limit", num: true, cell: (o) => o.limit_price ?? "--" },
          { header: "Req", num: true, cell: (o) => o.requested },
          { header: "Filled", num: true, cell: (o) => o.filled },
          { header: "Avg fill", num: true, cell: (o) => o.avg_fill_price ?? "--" },
          {
            header: "Slippage",
            num: true,
            cell: (o) => (o.slippage_cc === null ? "--" : `${o.slippage_cc}cc`),
          },
          { header: "Fees", num: true, cell: (o) => usd(o.fees_usd, 4) },
          {
            header: "Latency",
            num: true,
            cell: (o) => (o.submit_latency_ms ? `${o.submit_latency_ms.toFixed(1)}ms` : "--"),
          },
          { header: "Strategy", cell: (o) => o.strategy_id ?? "--" },
          {
            header: "Reject reason",
            cell: (o) => <span className="small">{o.reject_reason ?? ""}</span>,
          },
        ]}
      />
    </>
  );
}
