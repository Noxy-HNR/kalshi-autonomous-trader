/**
 * Typed client for the trading engine's API.
 *
 * Every call goes through `request`, which fails loudly rather than returning a
 * partial object. A dashboard that silently renders stale or empty data during
 * an outage is worse than one that says it cannot reach the engine: the whole
 * point of this surface is to tell you the truth about a system that is trading
 * your money.
 */

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly path: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail ?? detail;
    } catch {
      // A non-JSON error body is still an error; keep the status text.
    }
    throw new ApiError(detail, response.status, path);
  }
  return (await response.json()) as T;
}

export const get = <T,>(path: string) => request<T>(path);
export const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });

// --- shapes the engine returns ---------------------------------------------

export interface DomainSummary {
  available: boolean;
  strategies_active?: number;
  strategies_registered?: number;
  markets_watched?: number;
  open_orders?: number;
  open_positions?: number;
  cycles?: number;
  equity_usd?: string;
  balance_usd?: string;
  exposure_usd?: string;
  pending_exposure_usd?: string;
  realized_pnl_today_usd?: string;
  drawdown_pct?: string;
  /** Perpetuals only. */
  effective_leverage?: string | null;
  max_leverage?: string;
  exchange_account_leverage?: number | null;
  equity_source?: string;
  equity_is_known?: boolean;
  last_cycle?: CycleResult | null;
}

export interface CycleResult {
  markets_examined: number;
  markets_skipped_stale: number;
  intents_generated: number;
  intents_approved: number;
  intents_rejected: number;
  orders_submitted: number;
  errors: number;
  duration_ms: number;
  rejection_reasons: Record<string, number>;
}

export interface Overview {
  environment: string;
  operating_mode: string;
  live_armed: boolean;
  can_submit_real_orders: boolean;
  trading_permitted: boolean;
  phase: string;
  total_equity_usd: string;
  event: DomainSummary;
  perp: DomainSummary;
  health: Health;
}

export interface Health {
  phase?: string;
  trading_permitted?: boolean;
  database?: boolean;
  components?: ComponentHealth[];
  kill_switches?: KillSwitch[];
  writer?: { queued: number; capacity: number; written: number; dropped: number } | null;
  status?: string;
}

export interface ComponentHealth {
  name: string;
  healthy: boolean;
  detail: string;
  consecutive_failures: number;
}

export interface KillSwitch {
  scope: string;
  subject: string;
  engaged: boolean;
  reason: string | null;
  engaged_by?: string | null;
}

export interface LeveragePanel {
  available: boolean;
  max_leverage: string;
  equity_usd?: string;
  equity_source?: string;
  equity_is_known?: boolean;
  current_exposure_usd?: string;
  pending_exposure_usd?: string;
  current_leverage?: string | null;
  projected_leverage?: string | null;
  headroom_usd?: string;
  exchange_account_leverage?: number | null;
  exchange_leverage_note?: string;
  blocked_violations?: number;
  note?: string;
}

export interface ControlState {
  operating_mode: string;
  environment: string;
  live_armed: boolean;
  can_submit_real_orders: boolean;
  trading_permitted: boolean;
  phase: string;
  max_perp_leverage: string;
  kill_switches: KillSwitch[];
  modes: string[];
  scopes: string[];
}

export interface Decision {
  decision_id: string;
  ts: string | null;
  market_type: string;
  ticker: string;
  strategy_id: string | null;
  decision: string;
  reason_code: string | null;
  reason: string | null;
  requested: string;
  approved: string;
  market_price: string | null;
  fair_value: string | null;
  edge_cc: number | null;
  confidence: number | null;
  expected_value_usd: string | null;
  projected_leverage: string | null;
  order_id: string | null;
  environment: string;
}

export interface OrderRow {
  order_id: string;
  exchange_order_id: string | null;
  market_type: string;
  ticker: string;
  strategy_id: string | null;
  state: string;
  side: string | null;
  limit_price: string | null;
  requested: string;
  filled: string;
  avg_fill_price: string | null;
  fees_usd: string;
  slippage_cc: number | null;
  submit_latency_ms: number | null;
  reject_reason: string | null;
  environment: string;
  submitted_ts: string | null;
}

export interface MarketRow {
  ticker: string;
  market_type: string;
  event_ticker: string;
  status: string | null;
  bid: string | null;
  ask: string | null;
  mid: string | null;
  mark: string | null;
  spread_cc: number | null;
  depth_bid_qc: number;
  depth_ask_qc: number;
  volume_qc: number | null;
  open_interest_qc: number | null;
  age_ms: number | null;
  fresh: boolean;
}

export interface RiskEventRow {
  ts: string | null;
  /** Present on leverage rejections: the full arithmetic behind the refusal. */
  leverage?: Record<string, string> | null;
  market_type: string;
  ticker: string | null;
  strategy_id: string | null;
  reason_code: string;
  message: string | null;
  severity: string;
  limit_name: string | null;
  limit_value: string | null;
  observed_value: string | null;
  decision_id: string | null;
}

export interface AuditRow {
  ts: string | null;
  actor: string;
  action: string;
  subject: string | null;
  detail: Record<string, unknown> | null;
  severity: string;
  environment: string | null;
}

export interface StrategyRow {
  strategy_id: string;
  version?: string;
  name?: string;
  market_type?: string;
  family?: string;
  state: string;
  enabled?: boolean;
  confidence?: number | null;
  allocated_capital_usd?: string;
  /** Kept strictly separate; the dashboard never merges these. */
  train_metrics?: Record<string, unknown> | null;
  validation_metrics?: Record<string, unknown> | null;
  oos_metrics?: Record<string, unknown> | null;
  walk_forward_metrics?: Record<string, unknown> | null;
  robustness_metrics?: Record<string, unknown> | null;
  paper_metrics?: Record<string, unknown> | null;
  live_metrics?: Record<string, unknown> | null;
  retire_reason?: string | null;
  runtime?: { state: string; health?: Record<string, unknown> } | null;
}

export interface ModelRow {
  model_id: string;
  version: string;
  model_type: string;
  market_type: string;
  deployment_state: string;
  train_metrics: Record<string, number> | null;
  validation_metrics: Record<string, number> | null;
  test_metrics: Record<string, number> | null;
  live_metrics: Record<string, number> | null;
  calibration: Record<string, unknown> | null;
  git_commit: string | null;
  train_window?: (string | null)[];
  test_window?: (string | null)[];
}

export interface ArbRow {
  opportunity_id: string;
  ts: string | null;
  kind: string;
  market_type: string;
  legs: Array<Record<string, unknown>> | null;
  gross_edge_usd: string | null;
  fees_usd: string | null;
  net_edge_usd: string | null;
  expected_value_usd: string | null;
  worst_case_usd: string | null;
  execution_probability: number | null;
  actioned: boolean;
  reason: string | null;
}

export interface NewsRow {
  article_id: string;
  title: string;
  summary: string | null;
  url: string | null;
  source: string;
  published_ts: string | null;
  ingested_ts: string | null;
  source_reliability: number | null;
  relevance: number | null;
  entities: unknown[] | null;
  markets: Array<{ ticker: string; relevance: number; direction?: string | null }>;
}

export interface SystemStatus {
  process: Record<string, number | null>;
  host: Record<string, number>;
  market_data: Record<string, Record<string, unknown>>;
  metrics: {
    uptime_s: number;
    counters: Record<string, number>;
    gauges: Record<string, number>;
    histograms: Record<string, Record<string, number>>;
  };
  app: Record<string, unknown> | null;
}

// --- endpoints --------------------------------------------------------------

export interface Tunable {
  key: string;
  domain: string;
  label: string;
  description: string;
  plain: string;
  kind: string;
  value: string;
  default: string;
  minimum: string;
  maximum: string;
  step: string;
  higher_is_riskier: boolean;
  locked: boolean;
}

export interface TunableChange {
  key: string;
  previous: string;
  value: string;
  requested: string;
  clamped: boolean;
}

export const api = {
  overview: () => get<Overview>("/overview"),
  health: () => get<Health>("/health"),
  system: () => get<SystemStatus>("/system"),
  controlState: () => get<ControlState>("/control/state"),

  leverage: () => get<LeveragePanel>("/perps/leverage"),
  leverageViolations: () =>
    get<{ max_leverage: string; violations: RiskEventRow[] }>("/perps/violations"),
  perpPositions: () => get<{ available: boolean; positions: unknown[] }>("/perps/positions"),
  eventPositions: () => get<{ available: boolean; positions: unknown[] }>("/events/positions"),

  markets: (marketType?: string) =>
    get<{ markets: MarketRow[]; count: number }>(
      marketType ? `/markets?market_type=${marketType}` : "/markets",
    ),
  orderbook: (ticker: string) => get<Record<string, unknown>>(`/markets/${ticker}/orderbook`),

  orders: (state?: string) =>
    get<{ orders: OrderRow[] }>(state ? `/orders?state=${state}` : "/orders"),
  fills: () => get<{ fills: unknown[] }>("/fills"),
  decisions: (filter?: string) =>
    get<{ decisions: Decision[] }>(filter ? `/decisions?decision=${filter}` : "/decisions"),
  decision: (id: string) => get<Record<string, unknown>>(`/decisions/${id}`),

  riskEvents: () => get<{ events: RiskEventRow[] }>("/risk/events"),
  tunables: () => get<{ tunables: Tunable[] }>("/risk/tunables"),
  setTunable: (key: string, value: string, actor = "dashboard") =>
    post<TunableChange>("/risk/tunables", { key, value, actor }),
  riskLimits: () => get<Record<string, unknown>>("/risk/limits"),
  audit: (hours = 24) => get<{ entries: AuditRow[] }>(`/audit?hours=${hours}`),

  strategies: () => get<{ strategies: StrategyRow[] }>("/strategies"),
  strategyHistory: (id: string) =>
    get<{ transitions: unknown[] }>(`/strategies/${id}/history`),
  models: () => get<{ models: ModelRow[] }>("/models"),
  news: () => get<{ articles: NewsRow[] }>("/news"),
  arbitrage: () => get<{ opportunities: ArbRow[] }>("/arbitrage"),
  llmSignals: () => get<{ signals: unknown[] }>("/llm/signals"),

  // --- controls ---
  setMode: (mode: string) => post<{ operating_mode: string }>("/control/mode", { mode }),
  pause: () => post<{ paused: boolean }>("/control/pause"),
  resume: () => post<{ paused: boolean }>("/control/resume"),
  engageKillSwitch: (scope: string, reason: string, subject = "") =>
    post<{ scope: string; engaged: boolean }>("/control/kill-switch/engage", {
      scope,
      reason,
      subject,
    }),
  releaseKillSwitch: (scope: string, subject = "") =>
    post<{ scope: string; engaged: boolean }>("/control/kill-switch/release", {
      scope,
      subject,
    }),
  cancelAll: (marketType?: string) =>
    post<{ cancelled: number }>(
      marketType ? `/control/cancel-all?market_type=${marketType}` : "/control/cancel-all",
    ),
};

// --- formatting -------------------------------------------------------------

/** Render a dollar string from the API without inventing precision. */
export function usd(value: string | null | undefined, digits = 2): string {
  if (value === null || value === undefined) return "--";
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Centicents to a price string. 4200 -> "$0.4200". */
export function priceFromCc(cc: number | null | undefined): string {
  if (cc === null || cc === undefined) return "--";
  return `$${(cc / 10_000).toFixed(4)}`;
}

/** Centi-contracts to a contract count. 1000 -> "10". */
export function contracts(qc: number | null | undefined): string {
  if (qc === null || qc === undefined) return "--";
  return (qc / 100).toLocaleString("en-US", { maximumFractionDigits: 2 });
}

export function pct(value: string | number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) return "--";
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return `${(n * 100).toFixed(digits)}%`;
}

function magnitude(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

/**
 * Elapsed time, which says so when a timestamp is in the future.
 *
 * Clamping a negative elapsed time to zero renders a future timestamp as
 * "0s ago", indistinguishable from "just now", and hides a real data fault.
 * ESPN, for instance, stamps its August items `EST` when US Eastern is on
 * `EDT`, so every one of them arrives 59 minutes ahead of the moment it was
 * received. That is worth seeing rather than smoothing away.
 */
export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "--";
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return "--";
  const seconds = (Date.now() - then) / 1000;
  // A second or two either way is clock skew, not a fault.
  if (seconds < -2) return `${magnitude(-seconds)} ahead`;
  return `${magnitude(Math.max(0, seconds))} ago`;
}
