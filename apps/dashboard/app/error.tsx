"use client";

/**
 * The client error boundary.
 *
 * Without this file Next.js renders "Application error: a client-side exception
 * has occurred (see the browser console for more information)" — which tells the
 * reader nothing, offers no way forward, and on a trading dashboard is
 * indistinguishable from the platform itself being broken.
 *
 * The common case is not a bug at all
 * -----------------------------------
 * Next.js fingerprints its JavaScript bundles, so every rebuild changes their
 * filenames. A browser tab left open across a rebuild is still holding the old
 * filenames, and the first sidebar click after that asks for a chunk the server
 * no longer has. The request 404s, the import rejects, and the boundary catches
 * it.
 *
 * Nothing is wrong with the platform when that happens — the tab is simply a
 * build behind. The correct recovery is to fetch the new build, so that case
 * reloads itself rather than making someone work out that "client-side
 * exception" meant "press F5".
 *
 * The reload happens once
 * ----------------------
 * Guarded through sessionStorage. If the reload does not fix it, the second
 * occurrence shows the error rather than refreshing forever — a dashboard stuck
 * in a reload loop is worse than one showing a message, because it cannot even
 * be read.
 */

import { useEffect, useState } from "react";

const RELOAD_GUARD = "kat.chunk-reload-attempted";

/** Whether this is a stale-bundle failure rather than a fault in the app. */
function isStaleBuild(error: Error): boolean {
  const name = error.name ?? "";
  const message = error.message ?? "";
  return (
    name === "ChunkLoadError" ||
    /Loading chunk [\w-]+ failed/i.test(message) ||
    /Loading CSS chunk/i.test(message) ||
    // Safari and Firefox phrase a failed module fetch differently from Chrome.
    /error loading dynamically imported module/i.test(message) ||
    /Importing a module script failed/i.test(message)
  );
}

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const stale = isStaleBuild(error);
  const [reloading, setReloading] = useState(false);

  useEffect(() => {
    if (!stale) return;
    if (sessionStorage.getItem(RELOAD_GUARD)) return;
    sessionStorage.setItem(RELOAD_GUARD, "1");
    setReloading(true);
    // location.reload() would re-request the same stale document from the
    // browser cache. Replacing the URL forces a fresh document and with it the
    // current bundle filenames.
    window.location.replace(window.location.href);
  }, [stale]);

  useEffect(() => {
    // Reaching a render without an error means whatever went wrong is behind
    // us, so a later stale bundle is allowed its own single reload.
    if (!stale) sessionStorage.removeItem(RELOAD_GUARD);
  }, [stale]);

  if (reloading) {
    return (
      <div className="err">
        <strong>Loading the updated dashboard…</strong>
        <div className="small" style={{ marginTop: 6 }}>
          This tab was running an older build. Fetching the current one.
        </div>
      </div>
    );
  }

  return (
    <div className="err">
      <strong>
        {stale
          ? "This tab is running an outdated build."
          : "Something in the dashboard failed to render."}
      </strong>
      <div className="small" style={{ marginTop: 6 }}>
        {stale ? (
          <>
            The dashboard was rebuilt while this tab was open, so the page asked for
            files that no longer exist. Reloading fixes it. Nothing is wrong with the
            trading engine, and nothing about your positions or risk limits is affected
            by this — it is only the web page.
          </>
        ) : (
          <>
            This is a fault in the dashboard, not in the trading engine. The engine runs
            in its own process and is unaffected: it keeps trading, enforcing risk
            limits and recording decisions whether or not this page renders. Check{" "}
            <code className="mono">logs/kat.log</code> for what the engine is doing.
          </>
        )}
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <button onClick={() => reset()}>Try again</button>
        <button className="go" onClick={() => window.location.replace(window.location.href)}>
          Reload
        </button>
      </div>

      <details style={{ marginTop: 12 }}>
        <summary className="small" style={{ cursor: "pointer" }}>
          Technical detail
        </summary>
        <pre
          className="mono small"
          style={{ marginTop: 8, whiteSpace: "pre-wrap", wordBreak: "break-word" }}
        >
          {error.name}: {error.message}
          {error.digest ? `\ndigest: ${error.digest}` : ""}
        </pre>
      </details>
    </div>
  );
}
