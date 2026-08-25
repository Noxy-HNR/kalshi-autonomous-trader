"use client";

/**
 * The reference.
 *
 * Every control in the interface middle-clicks through to an anchor on this
 * page, so an entry's `anchor` is a link somebody is actually following:
 * renaming one breaks it.
 *
 * The content is served by the engine rather than written here, for two
 * reasons. The wording is then identical in the tooltip, the limits table and
 * this page, and each limit is documented with the value it *currently* has
 * rather than the value it shipped with. A reference describing settings you do
 * not have is worse than none.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type DocsModel, type Tunable } from "@/lib/api";
import { ErrorBox } from "@/components/ui";

/** Same unit rendering the tuning page uses, so a value reads identically. */
function display(t: Tunable, raw: string): string {
  const n = Number(raw);
  if (!Number.isFinite(n)) return raw;
  switch (t.kind) {
    case "percent":
      return `${(n * 100).toFixed(n < 0.01 ? 2 : 1)}%`;
    case "cents":
      return `${(n / 100).toFixed(2)}¢`;
    case "contracts": {
      const c = n / 100;
      return `${c.toFixed(c < 10 && c % 1 !== 0 ? 1 : 0)} contract${c === 1 ? "" : "s"}`;
    }
    case "usd":
      return `$${n.toFixed(2)}`;
    case "leverage":
      return `${n.toFixed(2)}×`;
    default:
      return String(n);
  }
}

/** Renders blank-line-separated prose as paragraphs. */
function Prose({ text }: { text: string }) {
  return (
    <>
      {text.split("\n\n").map((para, i) => (
        <p key={i} style={{ margin: "0 0 10px" }}>
          {para}
        </p>
      ))}
    </>
  );
}

function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} style={{ marginBottom: 34, scrollMarginTop: 20 }}>
      <h2 style={{ marginBottom: 4 }}>{title}</h2>
      {children}
    </section>
  );
}

export default function DocsPage() {
  const [docs, setDocs] = useState<DocsModel | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [query, setQuery] = useState("");
  const [highlight, setHighlight] = useState<string | null>(null);

  useEffect(() => {
    api.docs().then(setDocs).catch(setError);
  }, []);

  // Scroll to the anchor once content exists. The browser cannot do this
  // itself: the hash is present before the fetch resolves, so by the time the
  // target element is rendered the automatic scroll has already happened.
  const jump = useCallback(() => {
    const hash = window.location.hash.slice(1);
    if (!hash) return;
    const el = document.getElementById(hash);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    setHighlight(hash);
    window.setTimeout(() => setHighlight(null), 2400);
  }, []);

  useEffect(() => {
    if (!docs) return;
    jump();
    window.addEventListener("hashchange", jump);
    return () => window.removeEventListener("hashchange", jump);
  }, [docs, jump]);

  const filter = query.trim().toLowerCase();
  const matches = useCallback(
    (...parts: (string | undefined)[]) =>
      !filter || parts.some((p) => (p ?? "").toLowerCase().includes(filter)),
    [filter],
  );

  const concepts = useMemo(
    () => (docs?.concepts ?? []).filter((c) => matches(c.term, c.short, c.body)),
    [docs, matches],
  );
  const fields = useMemo(
    () => (docs?.risk_fields ?? []).filter((c) => matches(c.term, c.field, c.short, c.body)),
    [docs, matches],
  );
  const controls = useMemo(
    () => (docs?.controls ?? []).filter((c) => matches(c.term, c.short, c.body)),
    [docs, matches],
  );
  const tunables = useMemo(
    () =>
      (docs?.tunables ?? []).filter((t) =>
        matches(t.label, t.key, t.plain, t.raising, t.lowering, t.example),
      ),
    [docs, matches],
  );

  const card = (anchor: string) => ({
    scrollMarginTop: 20,
    marginBottom: 12,
    outline: highlight === anchor ? "2px solid var(--accent, #2f6fd0)" : undefined,
    transition: "outline .3s ease",
  });

  return (
    <>
      <h1>Reference</h1>
      <div className="sub">
        What every setting and control does, written for someone who has never traded.
        Middle-click any slider or control in the app to land on its entry here.
      </div>
      <ErrorBox error={error} />

      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search — try 'spread', 'leverage', 'drawdown'…"
        style={{
          width: "100%",
          padding: "9px 12px",
          marginBottom: 20,
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 7,
          color: "var(--text)",
          fontSize: 14,
        }}
      />

      {!docs ? (
        <div className="muted small">Loading…</div>
      ) : (
        <>
          {concepts.length ? (
            <Section id="concepts" title="The ideas">
              <div className="sub">
                Start here if any word in this app is unfamiliar. Nothing below assumes
                you have traded before.
              </div>
              {concepts.map((c) => (
                <div className="card" key={c.anchor} id={c.anchor} style={card(c.anchor)}>
                  <h3>{c.term}</h3>
                  <div className="sub" style={{ marginBottom: 8 }}>{c.short}</div>
                  <Prose text={c.body ?? ""} />
                </div>
              ))}
            </Section>
          ) : null}

          {tunables.length ? (
            <Section id="limits" title="The limits you can adjust">
              <div className="sub">
                Each of these is a slider on the Risk Tuning page. Values shown are what
                the platform is set to right now, not what it shipped with.
              </div>
              {tunables.map((t) => (
                <div className="card" key={t.anchor} id={t.anchor} style={card(t.anchor)}>
                  <div
                    className="row"
                    style={{ justifyContent: "space-between", alignItems: "baseline" }}
                  >
                    <h3 style={{ margin: 0 }}>{t.label}</h3>
                    <code className="small">{t.key}</code>
                  </div>

                  <div className="row small" style={{ gap: 14, margin: "8px 0 12px" }}>
                    <span>
                      now <strong>{display(t, t.value)}</strong>
                    </span>
                    <span className="muted">
                      default {display(t, t.default)}
                    </span>
                    <span className="muted">
                      range {display(t, t.minimum)} – {display(t, t.maximum)}
                    </span>
                  </div>

                  <Prose text={t.plain || t.description} />

                  {t.locked ? (
                    <div className="note">
                      This one has no slider and no number to change.
                    </div>
                  ) : (
                    <div className="grid g2" style={{ marginTop: 6 }}>
                      <div>
                        <div className="tune-label">RAISING IT</div>
                        <div className="small" style={{ marginTop: 6 }}>{t.raising}</div>
                      </div>
                      <div>
                        <div className="tune-label">LOWERING IT</div>
                        <div className="small" style={{ marginTop: 6 }}>{t.lowering}</div>
                      </div>
                    </div>
                  )}

                  {t.example ? (
                    <div className="note" style={{ marginTop: 12 }}>
                      <strong>For example:</strong> {t.example}
                    </div>
                  ) : null}
                </div>
              ))}
            </Section>
          ) : null}

          {fields.length ? (
            <Section id="other-limits" title="Limits set in configuration">
              <div className="sub">
                These appear in the Risk page&apos;s limits table. They are changed in{" "}
                <code className="mono">.env</code> rather than from the dashboard.
              </div>
              {fields.map((f) => (
                <div className="card" key={f.anchor} id={f.anchor} style={card(f.anchor)}>
                  <div
                    className="row"
                    style={{ justifyContent: "space-between", alignItems: "baseline" }}
                  >
                    <h3 style={{ margin: 0 }}>{f.term}</h3>
                    <code className="small">{f.field}</code>
                  </div>
                  <div className="sub" style={{ margin: "8px 0" }}>{f.short}</div>
                  <Prose text={f.body ?? ""} />
                </div>
              ))}
            </Section>
          ) : null}

          {controls.length ? (
            <Section id="controls" title="The controls">
              {controls.map((c) => (
                <div className="card" key={c.anchor} id={c.anchor} style={card(c.anchor)}>
                  <h3>{c.term}</h3>
                  <div className="sub" style={{ marginBottom: 8 }}>{c.short}</div>
                  <Prose text={c.body ?? ""} />
                </div>
              ))}
            </Section>
          ) : null}

          {!concepts.length && !tunables.length && !fields.length && !controls.length ? (
            <div className="empty">Nothing matches “{query}”.</div>
          ) : null}
        </>
      )}
    </>
  );
}
