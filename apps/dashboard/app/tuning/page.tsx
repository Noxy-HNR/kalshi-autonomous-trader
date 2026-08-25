"use client";

/**
 * Risk tuning.
 *
 * Every slider here changes a real limit on the running platform, immediately,
 * and every change is written to the audit log and persisted across restarts.
 * There is no "apply" button because there is no draft state: the value you see
 * is the value in force.
 *
 * The leverage ceiling appears in this list and has no slider. That is the
 * honest presentation -- not a disabled control implying a number someone could
 * enable, but an entry stating that the enforcement is a comparison between an
 * exposure and an equity with no limit parameter to set. See
 * kat/risk/invariants.py.
 *
 * Bounds come from the server, not from this file. A slider cannot be dragged
 * outside them and the API refuses an out-of-range value independently, so this
 * page is a convenience over the limits rather than the thing enforcing them.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Tunable } from "@/lib/api";
import { ErrorBox, Pill } from "@/components/ui";

/** Human-readable rendering of a raw value, by unit. */
function display(t: Tunable, raw: string): string {
  const n = Number(raw);
  if (!Number.isFinite(n)) return raw;
  switch (t.kind) {
    case "percent":
      return `${(n * 100).toFixed(n < 0.01 ? 2 : 1)}%`;
    case "cents":
      return `${(n / 100).toFixed(2)}¢`;
    case "contracts":
      return `${(n / 100).toFixed(0)} contracts`;
    case "usd":
      return `$${n.toFixed(2)}`;
    case "leverage":
      return `${n.toFixed(2)}×`;
    default:
      return String(n);
  }
}

/**
 * How risky this setting is, 0 = most conservative, 1 = most aggressive.
 *
 * Measured against the *shipped default*, not against position within the
 * min/max range. Several of these ranges are deliberately wide, so a perfectly
 * ordinary default can sit near a bound: judged by position, the default book
 * depth read as "aggressive" while being the value the platform ships with.
 *
 * The default anchors at 0.3 -- conservative, but not the floor, because the
 * defaults are cautious rather than maximally cautious and there is room to
 * tighten below them.
 */
const DEFAULT_LEVEL = 0.3;

function riskLevel(t: Tunable, raw: string): number {
  const v = Number(raw);
  const d = Number(t.default);
  const lo = Number(t.minimum);
  const hi = Number(t.maximum);
  if (!Number.isFinite(v) || !Number.isFinite(d)) return DEFAULT_LEVEL;
  if (v === d) return DEFAULT_LEVEL;

  const towardRisk = t.higher_is_riskier ? v > d : v < d;
  const riskyEnd = t.higher_is_riskier ? hi : lo;
  const safeEnd = t.higher_is_riskier ? lo : hi;

  if (towardRisk) {
    const span = Math.abs(riskyEnd - d) || 1;
    return DEFAULT_LEVEL + (1 - DEFAULT_LEVEL) * Math.min(1, Math.abs(v - d) / span);
  }
  const span = Math.abs(d - safeEnd) || 1;
  return DEFAULT_LEVEL * (1 - Math.min(1, Math.abs(d - v) / span));
}

function riskTone(level: number): "good" | "warn" | "bad" {
  if (level < 0.4) return "good";
  if (level < 0.75) return "warn";
  return "bad";
}

const GROUPS: { id: string; title: string; blurb: string; keys: RegExp }[] = [
  {
    id: "quality",
    title: "What counts as a tradeable market",
    blurb:
      "These decide which markets are worth entering at all. Loosening them does not " +
      "make the platform trade better — it makes it willing to trade worse markets.",
    keys: /(max_spread|min_book_depth|min_edge|max_slippage)/,
  },
  {
    id: "size",
    title: "How much may be at stake",
    blurb: "Concentration and sizing. These bound how much one mistake can cost.",
    keys: /(max_position|max_portfolio_exposure|max_exposure|max_order_notional|max_open_positions)/,
  },
  {
    id: "stop",
    title: "When to stop",
    blurb:
      "Loss budgets and rate limits. These are what turn a bad run into a pause " +
      "rather than an empty account.",
    keys: /(max_loss|max_daily_loss|max_drawdown|max_orders_per_minute)/,
  },
];

function groupOf(t: Tunable): string {
  return GROUPS.find((g) => g.keys.test(t.key))?.id ?? "stop";
}

export default function TuningPage() {
  const [tunables, setTunables] = useState<Tunable[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [domain, setDomain] = useState<"event" | "perp">("event");

  const load = useCallback(async () => {
    try {
      const res = await api.tunables();
      setTunables(res.tunables);
      setError(null);
      // Drop drafts that now match the server, so the page settles after a save.
      setDraft((d) => {
        const next = { ...d };
        for (const t of res.tunables) if (next[t.key] === t.value) delete next[t.key];
        return next;
      });
    } catch (e) {
      setError(e);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function commit(t: Tunable, value: string) {
    setBusy(t.key);
    setNote(null);
    try {
      const res = await api.setTunable(t.key, value);
      setNote(
        res.clamped
          ? `${t.label}: requested ${res.requested}, applied ${res.value} (clamped by the server)`
          : `${t.label}: ${display(t, res.previous)} → ${display(t, res.value)}`,
      );
      await load();
    } catch (e) {
      setNote(`${t.label}: refused — ${e instanceof Error ? e.message : String(e)}`);
      setDraft((d) => {
        const next = { ...d };
        delete next[t.key];
        return next;
      });
    } finally {
      setBusy(null);
    }
  }

  const shown = useMemo(
    () => tunables.filter((t) => t.domain === domain),
    [tunables, domain],
  );
  const locked = useMemo(() => tunables.filter((t) => t.locked), [tunables]);

  const overall = useMemo(() => {
    const live = shown.filter((t) => !t.locked);
    if (!live.length) return 0;
    return live.reduce((sum, t) => sum + riskLevel(t, draft[t.key] ?? t.value), 0) / live.length;
  }, [shown, draft]);

  return (
    <>
      <h1>Risk tuning</h1>
      <div className="sub">
        Every change here takes effect immediately on the running platform, is written to
        the audit log, and survives a restart. There is no apply button because there is
        no draft — what you see is what is in force.
      </div>
      <ErrorBox error={error} />

      <div className="row" style={{ gap: 8, margin: "12px 0" }}>
        {(["event", "perp"] as const).map((d) => (
          <button
            key={d}
            className={domain === d ? "go" : ""}
            onClick={() => setDomain(d)}
          >
            {d === "event" ? "Event contracts" : "Perpetuals"}
          </button>
        ))}
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div className="tune-label">OVERALL RISK POSTURE</div>
            <div className="sub" style={{ marginTop: 4 }}>
              How far these limits sit from the values the platform ships with.
              A rough summary, not a measurement of anything.
            </div>
          </div>
          <Pill tone={riskTone(overall)}>
            {overall < 0.4 ? "conservative" : overall < 0.75 ? "moderate" : "aggressive"}
          </Pill>
        </div>
        <div className="lev-bar" style={{ marginTop: 12 }}>
          <div
            className={`tune-fill ${riskTone(overall)}`}
            style={{ width: `${Math.round(overall * 100)}%` }}
          />
        </div>
      </div>

      {note ? (
        <div className="card" style={{ marginBottom: 16 }}>
          <span className="mono small">{note}</span>
        </div>
      ) : null}

      {GROUPS.map((g) => {
        const rows = shown.filter((t) => !t.locked && groupOf(t) === g.id);
        if (!rows.length) return null;
        return (
          <section key={g.id} style={{ marginBottom: 24 }}>
            <div className="tune-label">{g.title.toUpperCase()}</div>
            <div className="sub" style={{ margin: "4px 0 12px" }}>{g.blurb}</div>
            {rows.map((t) => {
              const value = draft[t.key] ?? t.value;
              const level = riskLevel(t, value);
              const dirty = value !== t.value;
              return (
                <div className="card" key={t.key} style={{ marginBottom: 12 }}>
                  <div
                    className="row"
                    style={{ justifyContent: "space-between", alignItems: "baseline" }}
                  >
                    <strong>{t.label}</strong>
                    <span className="row" style={{ gap: 8, alignItems: "center" }}>
                      <code style={{ fontSize: 15 }}>{display(t, value)}</code>
                      <Pill tone={riskTone(level)}>
                        {level < 0.4 ? "conservative" : level < 0.75 ? "moderate" : "aggressive"}
                      </Pill>
                    </span>
                  </div>
                  <div className="sub" style={{ margin: "6px 0 10px" }}>{t.description}</div>
                  <input
                    type="range"
                    min={t.minimum}
                    max={t.maximum}
                    step={t.step}
                    value={value}
                    disabled={busy === t.key}
                    onChange={(e) =>
                      setDraft((d) => ({ ...d, [t.key]: e.target.value }))
                    }
                    onMouseUp={(e) => commit(t, (e.target as HTMLInputElement).value)}
                    onKeyUp={(e) => {
                      if (e.key.startsWith("Arrow"))
                        commit(t, (e.target as HTMLInputElement).value);
                    }}
                    onTouchEnd={(e) => commit(t, (e.target as HTMLInputElement).value)}
                    style={{ width: "100%" }}
                  />
                  <div
                    className="row"
                    style={{ justifyContent: "space-between", fontSize: 12, opacity: 0.7 }}
                  >
                    <span>{display(t, t.minimum)}</span>
                    {dirty ? (
                      <span>saving…</span>
                    ) : value === t.default ? (
                      <span>default</span>
                    ) : (
                      <span>changed from {display(t, t.default)}</span>
                    )}
                    <span>{display(t, t.maximum)}</span>
                  </div>
                </div>
              );
            })}
          </section>
        );
      })}

      {locked.length ? (
        <section style={{ marginBottom: 24 }}>
          <div className="tune-label">NOT ADJUSTABLE</div>
          <div className="sub" style={{ margin: "4px 0 12px" }}>
            Shown here so it is visible beside the sliders rather than absent from the page.
          </div>
          {locked.map((t) => (
            <div className="card" key={t.key} style={{ marginBottom: 12, opacity: 0.95 }}>
              <div
                className="row"
                style={{ justifyContent: "space-between", alignItems: "baseline" }}
              >
                <strong>{t.label}</strong>
                <span className="row" style={{ gap: 8, alignItems: "center" }}>
                  <code style={{ fontSize: 18 }}>{display(t, t.value)}</code>
                  <Pill tone="good">locked</Pill>
                </span>
              </div>
              <div className="sub" style={{ marginTop: 6 }}>{t.description}</div>
            </div>
          ))}
        </section>
      ) : null}
    </>
  );
}
