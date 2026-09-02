import type { AgentInfo, Annotation, ExperimentRecord, ExperimentSummary } from "../api/types";
import { benchExperiments, normKey, type Benchmark } from "../data/benchmarks";
import { dbAgentFromExperiment } from "./dbIdentity";

export function agentTokens(agent: AgentInfo): string[] {
  return [agent.id, agent.label, ...(agent.aliases ?? [])].map(normKey).filter(Boolean);
}

export function findAgentExperiment(
  experiments: ExperimentSummary[],
  agent: AgentInfo,
): ExperimentSummary | undefined {
  if (agent.id === "claude-agent-sdk") return experiments[0];
  const tokens = agentTokens(agent);
  return experiments.find((e) => {
    const haystack = normKey(`${e.project_name ?? ""} ${e.dataset_name ?? ""}`);
    return tokens.some((t) => haystack.includes(t));
  });
}

export function benchDefaultSelection(
  b: Benchmark,
  experiments: ExperimentSummary[],
): { b: Benchmark; exp: ExperimentSummary; agent: AgentInfo | null } | null {
  const exps = benchExperiments(b, experiments);
  if (!exps.length) return null;
  for (const exp of exps) {
    const agent = dbAgentFromExperiment(exp);
    if (agent) return { b, exp, agent };
  }
  return { b, exp: exps[0], agent: null };
}

export function defaultSelection(
  experiments: ExperimentSummary[],
  benchmarks: Benchmark[],
): { b: Benchmark; exp: ExperimentSummary; agent: AgentInfo | null } | null {
  for (const b of benchmarks) {
    const sel = benchDefaultSelection(b, experiments);
    if (sel) return sel;
  }
  return null;
}

export function annotationAverage(records: ExperimentRecord[], name: string): number | null {
  const xs = records
    .map((record) => annotationScore(record, name))
    .filter((value): value is number => value != null);
  return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
}

export function annotationSum(records: ExperimentRecord[], name: string): number | null {
  const xs = records
    .map((record) => annotationScore(record, name))
    .filter((value): value is number => value != null);
  return xs.length ? xs.reduce((s, x) => s + x, 0) : null;
}

const LEGACY_METRIC_ALIASES: Record<string, string[]> = {
  correctness: ["correct", "accuracy", "resolved", "tb_resolved", "swe_resolved"],
  task_completion: ["task_succeeded"],
  wall_time: ["elapsed_time"],
};

export function metricAnnotation(record: ExperimentRecord, name: string): Annotation | undefined {
  const candidates = [name, ...(LEGACY_METRIC_ALIASES[name.toLowerCase()] ?? [])].map((candidate) =>
    candidate.toLowerCase(),
  );
  return candidates
    .map((candidate) =>
      (record.annotations ?? []).find((annotation) => String(annotation.name).toLowerCase() === candidate),
    )
    .find(Boolean);
}

export function annotationScore(record: ExperimentRecord, name: string): number | null {
  const score = metricAnnotation(record, name)?.score;
  return typeof score === "number" && Number.isFinite(score) ? score : null;
}

export function fmtMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatScore(value: number | null | undefined): string {
  return typeof value === "number" ? value.toFixed(2) : "—";
}

export function formatMetricValue(name: string, value: number | null | undefined): string {
  if (typeof value !== "number") return "—";
  if (name === "total_token_usage" || name === "total_token") {
    return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
  }
  if (name === "cost") return formatUsd(value);
  if (name === "wall_time") return `${value.toFixed(2)}s`;
  if (
    [
      "tool_call_count",
      "idle_turn_count",
      "tool_execution_error_rate",
      "repeated_tool_call_rate",
      "redcode_risky_operation_count",
    ].includes(name)
  ) {
    return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
  }
  return value.toFixed(2);
}

export function formatUsd(value: number): string {
  if (value === 0) return "$0";
  if (Math.abs(value) < 0.000001) return "$<0.000001";
  if (Math.abs(value) < 0.01) return `$${value.toFixed(6)}`;
  if (Math.abs(value) < 1) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

export function totalTokenUsage(records: ExperimentRecord[]): number | null {
  const values = records
    .map(recordTokenUsage)
    .filter((value): value is number => value != null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) : null;
}

export function recordTokenUsage(record: ExperimentRecord): number | null {
  const evaluated = annotationScore(record, "total_token_usage");
  if (evaluated != null) return evaluated;
  const hasPrompt = typeof record.prompt_token_count === "number";
  const hasCompletion = typeof record.completion_token_count === "number";
  if (hasPrompt || hasCompletion) {
    return (record.prompt_token_count ?? 0) + (record.completion_token_count ?? 0);
  }
  return null;
}

export function averageTokenUsage(records: ExperimentRecord[]): number | null {
  const values = records.map(recordTokenUsage).filter((value): value is number => value != null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

export function recordCost(record: ExperimentRecord): number | null {
  const evaluated = annotationScore(record, "cost");
  if (evaluated != null) return evaluated;
  return typeof record.calculated_cost === "number" ? record.calculated_cost : null;
}

export function averageRecordCost(records: ExperimentRecord[]): number | null {
  const values = records.map(recordCost).filter((value): value is number => value != null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}
