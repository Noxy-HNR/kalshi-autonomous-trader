"use client";

/**
 * The control panel.
 *
 * Requirement 20: the dashboard may change mode, pause and resume trading,
 * operate every kill switch, and cancel working orders -- but **no control here
 * can bypass a risk limit, a validation gate, or the 1.00x leverage ceiling**.
 * That is not enforced by this file; it is enforced by the fact that the API
 * exposes no endpoint capable of it and the execution engine accepts only a
 * risk-engine verdict. This page cannot do damage it is not allowed to do.
 *
 * Destructive actions ask for confirmation and require a written reason, which
 * goes into the audit log with the operator's name on it.
 */

import { useState } from "react";
import { api, type ControlState } from "@/lib/api";
import { ErrorBox, Help, Pill, usePoll } from "@/components/ui";
import { PAGE } from "@/lib/explainers";

const SCOPE_HELP: Record<string, string> = {
  global: "Stops all trading in both domains.",
  event: "Stops event-contract trading. Perpetuals continue.",
  perp: "Stops perpetual trading. Event contracts continue.",
  strategy: "Stops one named strategy.",
  risk: "Set automatically when a risk condition trips or reconciliation fails.",
};

export default function ControlsPage() {
  const { data, error, refresh } = usePoll<ControlState>(() => api.controlState(), 2000);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [reason, setReason] = useState("");

  async function act(label: string, fn: () => Promise<unknown>) {
    setBusy(label);
    setMessage(null);
    try {
      await fn();
      setMessage(`${label}: done`);
      refresh();
    } catch (e) {
      setMessage(`${label}: failed — ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  }

  const engaged = (scope: string) =>
    (data?.kill_switches ?? []).some((k) => k.scope === scope && k.engaged && !k.subject);

  return (
    <>
      <h1>
        Controls <Help wide>{PAGE.controls}</Help>
      </h1>
      <div className="sub">
        Operating mode, trading halt, and the kill switches. Every action is audited.
      </div>
      <ErrorBox error={error} />

      {message ? (
        <div className="card" style={{ marginBottom: 14 }}>
          <span className="mono small">{message}</span>
        </div>
      ) : null}

      {data ? (
        <>
          <div className="card" style={{ marginBottom: 14 }}>
            <h3>Current state</h3>
            <div className="row">
              <Pill tone={data.environment === "live" ? "bad" : "neutral"}>
                environment: {data.environment}
              </Pill>
              <Pill tone={data.can_submit_real_orders ? "bad" : "good"}>
                {data.can_submit_real_orders ? "REAL MONEY ARMED" : "no real orders"}
              </Pill>
              <Pill tone={data.trading_permitted ? "good" : "bad"}>
                {data.trading_permitted ? "trading permitted" : "trading halted"}
              </Pill>
              <Pill>{data.phase}</Pill>
              <Pill tone="good">max perp leverage {data.max_perp_leverage}&times;</Pill>
            </div>
            <div className="note">
              Live trading requires two independent settings: <code>KAT_ENVIRONMENT=live</code>{" "}
              and <code>KAT_LIVE_TRADING_ARMED=true</code>. Neither can be set from this
              dashboard &mdash; arming real money is deliberately a configuration change made
              at the machine, not a button.
            </div>
          </div>

          <h2>Operating mode</h2>
          <div className="card">
            <div className="row">
              {/* The list comes from the engine; an older or degraded response
                  may not carry it, and a missing list is an empty one, not a crash. */}
              {(data.modes ?? []).map((mode) => (
                <button
                  key={mode}
                  disabled={busy !== null || mode === data.operating_mode}
                  className={mode === data.operating_mode ? "go" : ""}
                  onClick={() => act(`set mode ${mode}`, () => api.setMode(mode))}
                >
                  {mode.replace(/_/g, " ")}
                  {mode === data.operating_mode ? " ✓" : ""}
                </button>
              ))}
            </div>
            <div className="note">
              Changing autonomy mode never changes a risk limit. <strong>gated discovery</strong>{" "}
              is the recommended default: strategies may be discovered and validated
              autonomously, but none reaches live capital without clearing every lifecycle gate.
            </div>
          </div>

          <h2>Trading halt</h2>
          <div className="card">
            <div className="row">
              <button
                className="danger"
                disabled={busy !== null || engaged("global")}
                onClick={() => act("pause trading", () => api.pause())}
              >
                Pause all trading
              </button>
              <button
                className="go"
                disabled={busy !== null || !engaged("global")}
                onClick={() => {
                  if (confirm("Resume trading in both domains?")) {
                    void act("resume trading", () => api.resume());
                  }
                }}
              >
                Resume trading
              </button>
              <button
                className="danger"
                disabled={busy !== null}
                onClick={() => {
                  if (confirm("Cancel every working order in both domains?")) {
                    void act("cancel all orders", () => api.cancelAll());
                  }
                }}
              >
                Cancel all working orders
              </button>
            </div>
            <div className="note">
              Cancelling only reduces risk, so it is always permitted regardless of halt state.
            </div>
          </div>

          <h2>Kill switches</h2>
          <div className="card">
            <div className="row" style={{ marginBottom: 12 }}>
              <input
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="Reason (recorded in the audit log)"
                style={{
                  flex: 1,
                  minWidth: 260,
                  padding: "6px 9px",
                  borderRadius: 5,
                  background: "var(--panel-2)",
                  color: "var(--text)",
                  border: "1px solid var(--border)",
                  font: "inherit",
                  fontSize: 12.5,
                }}
              />
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Scope</th>
                    <th>State</th>
                    <th>Reason</th>
                    <th>What it stops</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {["global", "event", "perp", "risk"].map((scope) => {
                    const state = (data.kill_switches ?? []).find(
                      (k) => k.scope === scope && !k.subject,
                    );
                    const on = Boolean(state?.engaged);
                    return (
                      <tr key={scope}>
                        <td className="mono">{scope}</td>
                        <td>
                          <Pill tone={on ? "bad" : "good"}>{on ? "engaged" : "clear"}</Pill>
                        </td>
                        <td className="small">{state?.reason ?? ""}</td>
                        <td className="small muted">{SCOPE_HELP[scope]}</td>
                        <td>
                          {on ? (
                            <button
                              className="go"
                              disabled={busy !== null}
                              onClick={() => {
                                if (confirm(`Release the ${scope} kill switch?`)) {
                                  void act(`release ${scope}`, () =>
                                    api.releaseKillSwitch(scope),
                                  );
                                }
                              }}
                            >
                              Release
                            </button>
                          ) : (
                            <button
                              className="danger"
                              disabled={busy !== null || !reason.trim()}
                              title={!reason.trim() ? "A reason is required" : ""}
                              onClick={() =>
                                act(`engage ${scope}`, () =>
                                  api.engageKillSwitch(scope, reason.trim()),
                                )
                              }
                            >
                              Engage
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="note">
              Kill switch state persists across restarts and fails closed: if it cannot be read
              at start-up, trading is treated as halted. Engaging takes effect in memory
              immediately and the database write is queued, so a halt works even when the
              database is the thing that is broken.
            </div>
          </div>
        </>
      ) : (
        !error && <div className="empty">Connecting to the trading engine…</div>
      )}
    </>
  );
}
