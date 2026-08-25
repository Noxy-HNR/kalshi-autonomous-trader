"use client";

/**
 * The Perpetuals tab.
 *
 * Requirement: MAX LEVERAGE 1.00x must be visually obvious. It is rendered
 * large, at the top, with an explicit note that it is architectural rather than
 * a setting -- because an operator reading this page needs to know not just what
 * the limit is but that nothing on this page can change it.
 *
 * The bar shows current and projected leverage against that ceiling, so the
 * question "how close am I, and where would I be if everything resting filled?"
 * is answered without arithmetic.
 */

import { api, usd, timeAgo, type LeveragePanel, type RiskEventRow } from "@/lib/api";
import { DataTable, ErrorBox, Help, Pill, Stat, usePoll } from "@/components/ui";
import { PAGE } from "@/lib/explainers";

interface PerpPosition {
  ticker: string;
  position: string;
  entry_price: string | null;
  mark_price: string | null;
  notional_usd: string;
  is_long: boolean;
}

export default function PerpsPage() {
  const { data: lev, error } = usePoll<LeveragePanel>(() => api.leverage(), 2000);
  const { data: positions } = usePoll<{ available: boolean; positions: PerpPosition[] }>(
    () => api.perpPositions() as Promise<{ available: boolean; positions: PerpPosition[] }>,
    3000,
  );
  const { data: violations } = usePoll<{ max_leverage: string; violations: RiskEventRow[] }>(
    () => api.leverageViolations(),
    5000,
  );

  const current = Number(lev?.current_leverage ?? 0);
  const projected = Number(lev?.projected_leverage ?? 0);
  const over = projected > 1;

  return (
    <>
      <h1>
        Perpetuals <Help wide>{PAGE.perps}</Help>
      </h1>
      <div className="sub">
        Margin, funding, exposure and the leverage ceiling this platform will not cross.
      </div>
      <ErrorBox error={error} />

      <div className="lev-panel">
        <div className="lev-max">
          <span className="label">Maximum leverage</span>
          <span className="value">{lev?.max_leverage ?? "1.00"}&times;</span>
          <span className="locked">
            architectural invariant — not a setting, and no control on this dashboard can
            raise it
          </span>
        </div>

        {lev?.available ? (
          <>
            <div className="grid g4" style={{ marginBottom: 14 }}>
              <Stat
                label="Current leverage"
                value={`${current.toFixed(4)}×`}
                tone={current > 1 ? "bad" : current > 0.8 ? "warn" : "good"}
                note="filled positions only"
                small
              />
              <Stat
                label="Projected leverage"
                value={`${projected.toFixed(4)}×`}
                tone={over ? "bad" : projected > 0.8 ? "warn" : "good"}
                note="if every resting order filled"
                small
              />
              <Stat
                label="Account equity"
                value={usd(lev.equity_usd)}
                note={
                  lev.equity_is_known
                    ? `source: ${lev.equity_source}`
                    : "UNKNOWN — all exposure-increasing orders blocked"
                }
                tone={lev.equity_is_known ? undefined : "bad"}
                small
              />
              <Stat
                label="Headroom"
                value={usd(lev.headroom_usd)}
                note="notional still available"
                small
              />
            </div>

            <div>
              <div className="row small muted" style={{ justifyContent: "space-between" }}>
                <span>0.00×</span>
                <span>exposure against equity</span>
                <span>1.00× ceiling</span>
              </div>
              <div className="lev-bar">
                <div
                  className={`lev-fill ${over ? "over" : projected > 0.8 ? "near" : "safe"}`}
                  style={{ width: `${Math.min(100, Math.max(0, projected * 100))}%` }}
                />
              </div>
              <div className="row small muted" style={{ justifyContent: "space-between" }}>
                <span>
                  current exposure {usd(lev.current_exposure_usd)} · pending{" "}
                  {usd(lev.pending_exposure_usd)}
                </span>
                <span>
                  blocked violations:{" "}
                  <strong style={{ color: (lev.blocked_violations ?? 0) > 0 ? "var(--warn)" : undefined }}>
                    {lev.blocked_violations ?? 0}
                  </strong>
                </span>
              </div>
            </div>

            {over ? (
              <div className="err" style={{ marginTop: 14, marginBottom: 0 }}>
                <strong>Leverage is above 1.00×.</strong> This can happen from a mark-price move
                with no order of ours involved. Exposure-increasing orders are blocked; orders
                that strictly reduce exposure remain permitted so the position can be brought
                back.
              </div>
            ) : null}

            <div className="note">
              Kalshi&apos;s own <code className="mono">account_leverage</code> is{" "}
              {lev.exchange_account_leverage ?? "not reported"} — that figure is notional divided
              by <em>maintenance margin</em>, not by equity, and would permit roughly ten times
              more real leverage. It is shown for cross-checking only and is never used as the
              limit.
            </div>
          </>
        ) : (
          <div className="empty">
            {lev?.note ?? "Perpetual account state is not available."}
          </div>
        )}
      </div>

      <h2>Open perpetual positions</h2>
      <DataTable
        rows={positions?.positions ?? []}
        rowKey={(row) => row.ticker}
        empty="No open perpetual positions."
        columns={[
          { header: "Market", cell: (r) => <span className="mono">{r.ticker}</span> },
          {
            header: "Side",
            cell: (r) => <Pill tone={r.is_long ? "good" : "warn"}>{r.is_long ? "long" : "short"}</Pill>,
          },
          { header: "Size", num: true, cell: (r) => r.position },
          { header: "Entry", num: true, cell: (r) => r.entry_price ?? "--" },
          { header: "Mark", num: true, cell: (r) => r.mark_price ?? "--" },
          { header: "Notional", num: true, cell: (r) => usd(r.notional_usd) },
        ]}
      />

      <h2>Blocked leverage violations</h2>
      <div className="sub small">
        Every order the invariant refused, with the arithmetic behind the refusal.
      </div>
      <DataTable
        rows={violations?.violations ?? []}
        rowKey={(_row, i) => String(i)}
        empty="No leverage violations have been blocked."
        columns={[
          { header: "When", cell: (r) => timeAgo(r.ts) },
          { header: "Market", cell: (r) => <span className="mono">{r.ticker ?? "--"}</span> },
          { header: "Strategy", cell: (r) => r.strategy_id ?? "--" },
          {
            header: "Projected",
            num: true,
            cell: (r) => (
              <span style={{ color: "var(--bad)" }}>
                {r.leverage?.projected_leverage ? `${r.leverage.projected_leverage}×` : "--"}
              </span>
            ),
          },
          { header: "Equity", num: true, cell: (r) => usd(r.leverage?.equity_usd) },
          {
            header: "Would be",
            num: true,
            cell: (r) => usd(r.leverage?.projected_exposure_usd),
          },
          { header: "Reason", cell: (r) => <span className="small">{r.reason_code}</span> },
        ]}
      />
    </>
  );
}
