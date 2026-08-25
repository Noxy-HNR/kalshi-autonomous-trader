"use client";

/**
 * The last-resort error boundary.
 *
 * `app/error.tsx` sits inside the root layout, so it cannot catch a failure in
 * the root layout itself. This one replaces the whole document when that
 * happens, which is why it renders its own <html> and <body> and cannot rely on
 * globals.css having loaded — every style here is inline on purpose.
 *
 * It should essentially never be seen. It exists because the alternative is a
 * blank white page on a surface whose entire job is to tell you the truth about
 * a system that is trading money.
 */

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          background: "#0a0d13",
          color: "#e6edf3",
          font: "14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 24,
        }}
      >
        <div style={{ maxWidth: 560 }}>
          <div style={{ fontSize: 18, fontWeight: 650, marginBottom: 10 }}>
            The dashboard could not start.
          </div>
          <p style={{ color: "#8b98a9", marginTop: 0 }}>
            This is a fault in the web interface, not in the trading engine. The engine
            runs as a separate process and is unaffected — it continues trading,
            enforcing every risk limit and recording every decision whether or not this
            page renders.
          </p>
          <p style={{ color: "#8b98a9" }}>
            To check on it without the dashboard:
          </p>
          <pre
            style={{
              background: "#11161f",
              border: "1px solid #1e2530",
              borderRadius: 6,
              padding: "10px 12px",
              overflowX: "auto",
              fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
              fontSize: 12.5,
            }}
          >
            curl http://127.0.0.1:8020/api/health
          </pre>
          <p style={{ color: "#8b98a9" }}>
            To stop trading entirely without the dashboard, engage the global kill
            switch:
          </p>
          <pre
            style={{
              background: "#11161f",
              border: "1px solid #1e2530",
              borderRadius: 6,
              padding: "10px 12px",
              overflowX: "auto",
              fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
              fontSize: 12.5,
            }}
          >
            {`curl -X POST http://127.0.0.1:8020/api/control/kill-switch/engage \\
  -H "Content-Type: application/json" \\
  -d '{"scope":"global","reason":"dashboard down","actor":"operator"}'`}
          </pre>

          <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
            <button
              onClick={() => reset()}
              style={{
                background: "#1e2530",
                color: "#e6edf3",
                border: "1px solid #2a3341",
                borderRadius: 6,
                padding: "7px 14px",
                cursor: "pointer",
              }}
            >
              Try again
            </button>
            <button
              onClick={() => window.location.replace(window.location.href)}
              style={{
                background: "#2f6fd0",
                color: "#fff",
                border: "1px solid #2f6fd0",
                borderRadius: 6,
                padding: "7px 14px",
                cursor: "pointer",
              }}
            >
              Reload
            </button>
          </div>

          <pre
            style={{
              marginTop: 16,
              color: "#8b98a9",
              fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
              fontSize: 12,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {error.name}: {error.message}
            {error.digest ? `\ndigest: ${error.digest}` : ""}
          </pre>
        </div>
      </body>
    </html>
  );
}
