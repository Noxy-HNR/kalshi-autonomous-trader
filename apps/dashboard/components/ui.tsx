"use client";

/**
 * Shared presentation pieces.
 *
 * The bias throughout: show the real number, show when it is unknown, and never
 * render a placeholder that could be mistaken for data. A dashboard for a system
 * that trades money has to be legible under stress, so state is communicated by
 * colour *and* by words, never by colour alone.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";

/**
 * A hover explainer.
 *
 * Written for someone who has never traded. The rule followed throughout: say
 * what the number *is* in plain words, then what it means for them, and never
 * assume a term has been met before. "Spread" is not common knowledge; neither
 * is "drawdown", "notional", "fill" or "edge".
 *
 * Rendered as a real focusable element rather than a bare `title` attribute so
 * it opens on keyboard focus too, and carries `title` as well so it still works
 * if CSS fails to load.
 */
export function Help({ children, wide }: { children: React.ReactNode; wide?: boolean }) {
  const text = typeof children === "string" ? children : undefined;
  return (
    <span className="help" tabIndex={0} role="note" aria-label={text} title={text}>
      <span className="help-mark" aria-hidden="true">
        ?
      </span>
      <span className={`help-bubble${wide ? " wide" : ""}`}>{children}</span>
    </span>
  );
}

/**
 * A page heading with an explainer attached.
 *
 * Every page gets one. Someone landing on "Arbitrage" or "Walk-forward" with no
 * background should be able to read the first two lines and know what they are
 * looking at and whether it matters.
 */
export function PageIntro({
  title,
  lede,
  explain,
}: {
  title: string;
  lede: string;
  explain: React.ReactNode;
}) {
  return (
    <>
      <h1>
        {title} <Help wide>{explain}</Help>
      </h1>
      <div className="sub">{lede}</div>
    </>
  );
}

export function Stat({
  label,
  value,
  tone,
  note,
  small,
  explain,
}: {
  label: string;
  value: string;
  tone?: "good" | "bad" | "warn";
  note?: string;
  small?: boolean;
  explain?: React.ReactNode;
}) {
  return (
    <div className="card">
      <h3>
        {label}
        {explain ? <Help>{explain}</Help> : null}
      </h3>
      <div className={`stat${small ? " sm" : ""}${tone ? ` ${tone}` : ""}`}>{value}</div>
      {note ? <div className="stat-note">{note}</div> : null}
    </div>
  );
}

export function Pill({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "good" | "warn" | "bad" | "neutral";
}) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

/** Map an order/strategy/deployment state to a tone, defaulting to neutral. */
export function stateTone(state: string): "good" | "warn" | "bad" | "neutral" {
  const s = state.toLowerCase();
  if (["filled", "live", "deployed", "acknowledged", "healthy", "running", "ok"].includes(s))
    return "good";
  if (
    ["rejected", "risk_rejected", "retired", "failed", "unhealthy", "cancelled", "expired"].includes(
      s,
    )
  )
    return "bad";
  if (["submitted", "unknown", "partially_filled", "probation", "paper", "degraded"].includes(s))
    return "warn";
  return "neutral";
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const message =
    error instanceof ApiError
      ? `${error.message} (${error.status} on ${error.path})`
      : error instanceof Error
        ? error.message
        : String(error);
  return (
    <div className="err">
      <strong>Cannot reach the trading engine.</strong> {message}
      <div className="small" style={{ marginTop: 6 }}>
        The engine serves this API in its own process. If it is not running, start it with{" "}
        <code className="mono">python -m kat.engine.entrypoint</code>. Nothing below is live
        while this message is showing.
      </div>
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="empty">{children}</div>;
}

/**
 * Poll an endpoint on an interval.
 *
 * Returns the previous value while refreshing so the panel does not flash, but
 * surfaces `error` immediately -- stale data shown as if it were current is the
 * failure mode this whole surface exists to avoid.
 */
export function usePoll<T>(
  fetcher: () => Promise<T>,
  intervalMs = 3000,
): { data: T | null; error: unknown; loading: boolean; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  // Callers build a new `fetcher` closure on every render, so depending on it
  // directly would tear the interval down and restart it each time. It is held
  // in a ref instead: the interval stays stable while the function it calls is
  // always the current one rather than the one captured on first render.
  const latest = useRef(fetcher);
  latest.current = fetcher;

  const run = useCallback(async () => {
    try {
      const result = await latest.current();
      setData(result);
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      try {
        const result = await latest.current();
        // Re-checked after the await: navigating away mid-request would
        // otherwise let a resolved response write into a component that is no
        // longer mounted, and on a fast click-through that response belongs to
        // the page you just left.
        if (!alive) return;
        setData(result);
        setError(null);
      } catch (e) {
        if (alive) setError(e);
      } finally {
        if (alive) setLoading(false);
      }
    };
    void tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { data, error, loading, refresh: run };
}

export function DataTable<T>({
  rows,
  columns,
  empty = "Nothing to show yet.",
  rowKey,
}: {
  rows: T[];
  columns: Array<{
    header: string;
    cell: (row: T) => React.ReactNode;
    num?: boolean;
  }>;
  empty?: string;
  rowKey: (row: T, index: number) => string;
}) {
  if (!rows.length) return <Empty>{empty}</Empty>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.header} className={c.num ? "num" : undefined}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={rowKey(row, i)}>
              {columns.map((c) => (
                <td key={c.header} className={c.num ? "num" : undefined}>
                  {c.cell(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The environment banner.
 *
 * Requirement 43: when real orders can be submitted the dashboard must clearly
 * display ENVIRONMENT: LIVE. Live is styled unmistakably differently -- a red
 * pulsing border and explicit "REAL MONEY" text -- because the single most
 * dangerous mistake an operator can make is believing they are in paper mode.
 */
export function EnvironmentBanner({
  environment,
  canSubmitRealOrders,
  tradingPermitted,
  phase,
  mode,
}: {
  environment: string;
  canSubmitRealOrders: boolean;
  tradingPermitted: boolean;
  phase: string;
  mode: string;
}) {
  const env = environment.toLowerCase();
  const cls = env === "live" ? "env-live" : env === "demo" ? "env-demo" : "env-paper";
  return (
    <div className={`env-banner ${cls}`}>
      <span>
        ENVIRONMENT: {environment.toUpperCase()}
        {canSubmitRealOrders ? "  —  REAL MONEY AT RISK" : ""}
        {env === "paper" ? "  —  simulated fills, no orders sent" : ""}
        {env === "demo" ? "  —  real orders, Kalshi demo exchange" : ""}
      </span>
      <span className="row small">
        <Pill tone={tradingPermitted ? "good" : "bad"}>
          {tradingPermitted ? "trading permitted" : "trading halted"}
        </Pill>
        <Pill tone={phase === "running" ? "good" : phase === "degraded" ? "warn" : "neutral"}>
          {phase}
        </Pill>
        <Pill>{mode.replace(/_/g, " ")}</Pill>
      </span>
    </div>
  );
}
