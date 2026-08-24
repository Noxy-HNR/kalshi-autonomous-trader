"use client";

/**
 * Model registry.
 *
 * Train, validation and test metrics are separate columns and are never blended.
 * Selection happens on the test column only; the train column is shown so the
 * overfitting gap between them is visible rather than hidden.
 */

import { api, type ModelRow } from "@/lib/api";
import { DataTable, ErrorBox, Pill, stateTone, usePoll } from "@/components/ui";

function num(block: Record<string, number> | null | undefined, key: string, digits = 4): string {
  const v = block?.[key];
  return v === null || v === undefined || !Number.isFinite(v) ? "--" : v.toFixed(digits);
}

export default function ModelsPage() {
  const { data, error } = usePoll<{ models: ModelRow[] }>(() => api.models(), 5000);
  const models = data?.models ?? [];

  return (
    <>
      <h1>Models</h1>
      <div className="sub">
        Versions, training windows, per-split metrics, calibration and deployment state.
      </div>
      <ErrorBox error={error} />

      <DataTable
        rows={models}
        rowKey={(m) => `${m.model_id}:${m.version}`}
        empty="No models registered yet. The trainer registers a model once it clears every out-of-sample gate."
        columns={[
          { header: "Model", cell: (m) => <span className="mono small">{m.model_id}</span> },
          { header: "Ver", cell: (m) => <span className="mono small">{m.version}</span> },
          { header: "Family", cell: (m) => m.model_type },
          { header: "Domain", cell: (m) => <Pill>{m.market_type}</Pill> },
          {
            header: "State",
            cell: (m) => <Pill tone={stateTone(m.deployment_state)}>{m.deployment_state}</Pill>,
          },
          { header: "Train AUC", num: true, cell: (m) => num(m.train_metrics, "auc") },
          { header: "Val AUC", num: true, cell: (m) => num(m.validation_metrics, "auc") },
          { header: "Test AUC", num: true, cell: (m) => num(m.test_metrics, "auc") },
          { header: "Test Brier", num: true, cell: (m) => num(m.test_metrics, "brier") },
          { header: "Test ECE", num: true, cell: (m) => num(m.test_metrics, "ece") },
          {
            header: "Excess ECE",
            num: true,
            cell: (m) => num(m.test_metrics, "excess_ece"),
          },
          {
            header: "Overfit gap",
            num: true,
            cell: (m) => {
              const train = m.train_metrics?.auc;
              const test = m.test_metrics?.auc;
              if (train === undefined || test === undefined) return "--";
              const gap = train - test;
              return (
                <span style={{ color: gap > 0.15 ? "var(--bad)" : undefined }}>
                  {gap.toFixed(4)}
                </span>
              );
            },
          },
          {
            header: "Calibrator",
            cell: (m) => String((m.calibration as Record<string, unknown>)?.method ?? "--"),
          },
          { header: "Commit", cell: (m) => <span className="mono small">{m.git_commit ?? "--"}</span> },
        ]}
      />

      <div className="note">
        Excess ECE is the calibration error left after subtracting what binomial sampling noise
        explains at that sample size. It is the number with economic meaning: a systematic
        two-point error manufactures a two-cent edge on a $0.50 market, whereas noise averages
        out across trades.
      </div>
    </>
  );
}
