"use client";

/**
 * Small inline-SVG chart primitives.
 *
 * No charting library: every value drawn here comes straight from the same
 * polls that feed the tables elsewhere in the dashboard. `useSeries` builds a
 * time series out of real observations as they arrive (persisted per-browser
 * in localStorage so it survives a reload) rather than inventing a history —
 * a system trading real money doesn't get a demo trendline. A chart with
 * fewer than two points says so and waits, instead of drawing a line through
 * nothing.
 */

import { useEffect, useId, useRef, useState } from "react";

export interface SeriesPoint {
  t: number;
  v: number;
}

/**
 * Appends `extract(source)` to a persisted series whenever `source` changes
 * (i.e. once per successful poll). `extract` is read through a ref, mirroring
 * `usePoll`'s pattern, so callers can pass an inline closure without
 * retriggering the effect on every render.
 */
export function useSeries<T>(
  key: string,
  source: T | null | undefined,
  extract: (s: T) => number | null | undefined,
  max = 300,
): SeriesPoint[] {
  const [series, setSeries] = useState<SeriesPoint[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      const raw = window.localStorage.getItem(key);
      const parsed = raw ? (JSON.parse(raw) as unknown) : [];
      return Array.isArray(parsed) ? (parsed as SeriesPoint[]) : [];
    } catch {
      return [];
    }
  });

  const extractRef = useRef(extract);
  extractRef.current = extract;

  useEffect(() => {
    if (!source) return;
    const v = extractRef.current(source);
    if (v === null || v === undefined || !Number.isFinite(v)) return;
    setSeries((prev) => {
      const next = [...prev, { t: Date.now(), v }].slice(-max);
      try {
        window.localStorage.setItem(key, JSON.stringify(next));
      } catch {
        // Private browsing or a full quota just means local history resets.
      }
      return next;
    });
    // `source` changing reference is the poll tick; that's the only trigger we want.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, key, max]);

  return series;
}

/** A filled line chart. Reads real observed points, never interpolated ones. */
export function AreaChart({
  points,
  height = 84,
  color = "var(--accent)",
  format = (v: number) => v.toFixed(2),
  emptyLabel = "Collecting live data — this fills in as the dashboard stays open.",
}: {
  points: SeriesPoint[];
  height?: number;
  color?: string;
  format?: (v: number) => string;
  emptyLabel?: string;
}) {
  const gid = useId().replace(/[^a-zA-Z0-9]/g, "");

  if (points.length < 2) {
    return (
      <div className="chart-empty" style={{ height }}>
        {emptyLabel}
      </div>
    );
  }

  const W = 600;
  const H = height;
  const pad = 4;
  const values = points.map((p) => p.v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.abs(min) || 1;
  const x = (i: number) => (i / (points.length - 1)) * (W - pad * 2) + pad;
  const y = (v: number) => H - pad - ((v - min) / span) * (H - pad * 2);
  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(p.v).toFixed(2)}`)
    .join(" ");
  const areaPath = `${linePath} L${x(points.length - 1).toFixed(2)},${H} L${x(0).toFixed(2)},${H} Z`;
  const first = points[0].v;
  const last = points[points.length - 1].v;
  const trend = last >= first ? "good" : "bad";

  return (
    <div className="chart">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="chart-svg"
        style={{ height }}
      >
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.35" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill={`url(#${gid})`} stroke="none" />
        <path
          d={linePath}
          fill="none"
          stroke={color}
          strokeWidth={1.75}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="chart-range row small muted" style={{ justifyContent: "space-between" }}>
        <span>{format(min)}</span>
        <span className={trend}>{format(last)} now</span>
        <span>{format(max)}</span>
      </div>
    </div>
  );
}

/** A donut allocation chart built from SVG stroke arcs — no path trig required. */
export function DonutChart({
  segments,
  size = 118,
  thickness = 15,
  centerLabel,
  centerValue,
  emptyLabel = "Nothing allocated yet.",
}: {
  segments: { label: string; value: number; color: string }[];
  size?: number;
  thickness?: number;
  centerLabel?: string;
  centerValue?: string;
  emptyLabel?: string;
}) {
  const positive = segments.filter((s) => s.value > 0);
  const total = positive.reduce((sum, s) => sum + s.value, 0);
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  let offset = 0;

  return (
    <div className="donut-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--border)"
          strokeWidth={thickness}
        />
        {total > 0 &&
          positive.map((seg) => {
            const dash = (seg.value / total) * c;
            const el = (
              <circle
                key={seg.label}
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={seg.color}
                strokeWidth={thickness}
                strokeDasharray={`${dash} ${c - dash}`}
                strokeDashoffset={-offset}
                transform={`rotate(-90 ${size / 2} ${size / 2})`}
              />
            );
            offset += dash;
            return el;
          })}
        {centerValue ? (
          <text x="50%" y="47%" textAnchor="middle" className="donut-value" fill="var(--text)">
            {centerValue}
          </text>
        ) : null}
        {centerLabel ? (
          <text x="50%" y="63%" textAnchor="middle" className="donut-label" fill="var(--muted)">
            {centerLabel}
          </text>
        ) : null}
      </svg>
      <div className="donut-legend">
        {total > 0 ? (
          positive.map((seg) => (
            <div key={seg.label} className="row small" style={{ gap: 6 }}>
              <span className="swatch" style={{ background: seg.color }} />
              <span className="muted">{seg.label}</span>
              <span className="mono" style={{ marginLeft: "auto" }}>
                {((seg.value / total) * 100).toFixed(1)}%
              </span>
            </div>
          ))
        ) : (
          <div className="muted small">{emptyLabel}</div>
        )}
      </div>
    </div>
  );
}

/** Horizontal bars, sized against the largest magnitude in the set. */
export function HBarChart({
  rows,
  format = (v: number) => v.toFixed(2),
  emptyLabel = "Nothing to show yet.",
}: {
  rows: { label: string; value: number; tone?: "good" | "bad" | "warn" | "accent" }[];
  format?: (v: number) => string;
  emptyLabel?: string;
}) {
  if (!rows.length) return <div className="chart-empty">{emptyLabel}</div>;
  const max = Math.max(1e-9, ...rows.map((r) => Math.abs(r.value)));
  return (
    <div className="hbar-chart">
      {rows.map((r) => (
        <div key={r.label} className="hbar-row">
          <span className="hbar-label small muted" title={r.label}>
            {r.label}
          </span>
          <div className="hbar-track">
            <div
              className={`hbar-fill ${r.tone ?? "accent"}`}
              style={{ width: `${Math.min(100, (Math.abs(r.value) / max) * 100)}%` }}
            />
          </div>
          <span className="hbar-value mono small">{format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}
