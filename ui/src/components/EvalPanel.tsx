import type { AgentInfo, ExperimentContext, ExperimentRecord } from "../api/types";
import { CATS, type Benchmark } from "../data/benchmarks";
import {
  annotationAverage,
  averageRecordCost,
  formatMetricValue,
  totalTokenUsage,
} from "../utils/eval";
import { getMetricDescription, metricRangeClass } from "../utils/metricDescriptions";
import { FishboneCard } from "./FishboneCard";
import { MetricTooltip } from "./MetricTooltip";

const EFFICIENCY_METRICS = ["conciseness", "total_token_usage", "cost", "turn_count", "elapsed_time"] as const;
const SAFETY_METRICS = [
  "hallucination",
  "privacy_leakage",
  "unauthorized_action",
  "harmful_action",
  "failure_transparency",
  "prompt_injection_resilience",
] as const;
const ACCURACY_METRICS = [
  "correctness",
  "task_succeeded",
] as const;

interface Props {
  benchmark: Benchmark | null;
  records: ExperimentRecord[];
  context: ExperimentContext | null;
  agent: AgentInfo | null;
  experimentDatasetName?: string;
  projectName?: string;
  testedAgentModel?: string;
  judgeModel?: string;
}

function infoItem(label: string, value: string) {
  return (
    <div className="info-item">
      <div className="info-label">{label}</div>
      <div className="info-value">{value || "—"}</div>
    </div>
  );
}

export function EvalPanel({
  benchmark,
  records,
  context,
  agent,
  experimentDatasetName,
  projectName,
  testedAgentModel,
  judgeModel,
}: Props) {
  if (!benchmark || !records.length) {
    return (
      <article className="panel eval">
        <div className="panel-inner" id="eval-body">
          <p className="kicker">Eval</p>
          <p className="muted">← Select a benchmark in Task</p>
        </div>
      </article>
    );
  }

  const names: string[] = [];
  for (const r of records) {
    for (const a of r.annotations ?? []) {
      if (!names.includes(a.name)) names.push(a.name);
    }
  }

  const avgOf = (name: string) => {
    const target = name.toLowerCase();
    const xs = records
      .map((r) => (r.annotations ?? []).find((a) => String(a.name).toLowerCase() === target)?.score)
      .filter((x): x is number => typeof x === "number");
    return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
  };

  const avgFirst = (candidates: string[]) => {
    for (const name of candidates) {
      const value = avgOf(name);
      if (value != null) return value;
    }
    return null;
  };

  const overall = avgFirst(["correctness", "task_succeeded"]);
  const totalToken = totalTokenUsage(records);
  const hasScore = typeof overall === "number";
  const pct = hasScore ? Math.max(0, Math.min(100, overall * 100)) : 0;
  const good = hasScore && overall >= 0.5;
  const overallDescription = [
    "- Meaning: Benchmark-level average correctness score used as the overall result.",
    "- Calculation: Average of the first available correctness-style metric: correctness or task_succeeded.",
    "- Display: 0 to 1; higher is better.",
  ].join("\n");
  const totalTokenDescription = [
    "- Meaning: Total token usage across the selected benchmark.",
    "- Calculation: Uses sample prompt and completion tokens first, then falls back to the total_token_usage annotation.",
    "- Display: Non-negative token count.",
  ].join("\n");
  const domain =
    (benchmark.cat != null ? CATS[benchmark.cat] : undefined) ||
    context?.inputs?.domains?.join(", ") ||
    benchmark.name;

  const metricValue = (name: string): number | null => {
    if (name === "total_token_usage") return totalTokenUsage(records);
    if (name === "cost") return averageRecordCost(records);
    return annotationAverage(records, name);
  };

  return (
    <article className="panel eval">
      <div className="panel-inner" id="eval-body">
        <p className="kicker">Eval · {benchmark.name} all-sample average</p>

        <div className="card eval-summary-card">
          <p className="card-label">SUMMARY</p>
          <div className="eval-summary-main">
            <div
              className={`ring summary-ring ${hasScore ? (good ? "good" : "bad") : ""}`}
              style={{ ["--p" as string]: pct }}
            >
              {hasScore ? overall.toFixed(2) : "—"}
            </div>
            <div className="summary-copy">
              <div className="summary-title has-metric-tooltip">
                Overall evaluator score
                <MetricTooltip text={overallDescription} />
              </div>
              <div className="summary-sub">
                {records.length} samples · average correctness
              </div>
            </div>
            <div className="summary-token has-metric-tooltip">
              <span>total_token</span>
              <strong>{formatMetricValue("total_token", totalToken)}</strong>
              <MetricTooltip text={totalTokenDescription} />
            </div>
          </div>
          <details className="summary-config">
            <summary>Configuration</summary>
            <div className="eval-info-card compact embedded">
              <div className="info-grid">
                {infoItem("Agent", agent?.label ?? context?.agent?.names?.join(", ") ?? "—")}
                {infoItem("Tested agent model", testedAgentModel || "—")}
                {infoItem("LLM-as-a-judge model", judgeModel || "—")}
                {infoItem("Benchmark", benchmark.name)}
                {infoItem("Dataset", context?.dataset?.name ?? experimentDatasetName ?? "—")}
                {infoItem("Project ID", projectName ?? context?.experiment?.project_name ?? "—")}
                {infoItem("Domain", domain)}
                {infoItem(
                  "Runs OK",
                  context?.runs?.ok_runs != null
                    ? `${context.runs.ok_runs}/${records.length}`
                    : "—",
                )}
              </div>
            </div>
          </details>
        </div>

        <FishboneCard records={records} overall={overall} />

        <div className="card eval-assessment-card">
          <p className="card-label">EVALUATION</p>
          <div className="assessment-tree">
            <div className="assessment-root">
              <span>Eval Tree</span>
              <strong>Metrics</strong>
            </div>
            {[
              ["Correctness", "correctness", ACCURACY_METRICS.map((name) => [name, metricValue(name)] as const)],
              ["Safety", "safety", SAFETY_METRICS.map((name) => [name, metricValue(name)] as const)],
              ["Efficiency", "efficiency", EFFICIENCY_METRICS.map((name) => [name, metricValue(name)] as const)],
            ].map(([label, group, metrics], i) => (
              <div key={String(label)} className={`assessment-row assessment-${group}`}>
                <div className="assessment-label">
                  <span>{String(i + 1).padStart(2, "0")}</span>
                  <strong>{String(label)}</strong>
                </div>
                <div className="assessment-values">
                  {(metrics as [string, number | null][]).map(([name, value]) => {
                    const description = getMetricDescription(name, value);
                    const rangeClass = metricRangeClass(name, value);
                    return (
                      <div
                        key={name}
                        className={`assessment-metric has-metric-tooltip ${rangeClass}`}
                      >
                        <span>{name}</span>
                        <strong>{formatMetricValue(name, value)}</strong>
                        <MetricTooltip text={description} />
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </article>
  );
}
