import type { AgentInfo, ExperimentRecord, ExperimentSummary } from "../api/types";
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
  const target = name.toLowerCase();
  const xs = records
    .map((r) => (r.annotations ?? []).find((a) => String(a.name).toLowerCase() === target)?.score)
    .filter((x): x is number => typeof x === "number");
  return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
}

export function annotationSum(records: ExperimentRecord[], name: string): number | null {
  const target = name.toLowerCase();
  const xs = records
    .map((r) => (r.annotations ?? []).find((a) => String(a.name).toLowerCase() === target)?.score)
    .filter((x): x is number => typeof x === "number");
  return xs.length ? xs.reduce((s, x) => s + x, 0) : null;
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
  if (name === "total_token_usage" || name === "total_token" || name === "answer_cost") {
    return Math.round(value).toLocaleString("en-US");
  }
  if (name === "cost") return formatUsd(value);
  if (name === "tool_call_count") return (Math.round(value * 10) / 10).toLocaleString("en-US");
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
  const hasPrompt = typeof record.prompt_token_count === "number";
  const hasCompletion = typeof record.completion_token_count === "number";
  if (hasPrompt || hasCompletion) {
    return (record.prompt_token_count ?? 0) + (record.completion_token_count ?? 0);
  }
  const annotation = (record.annotations ?? []).find(
    (item) => String(item.name).toLowerCase() === "total_token_usage",
  );
  return typeof annotation?.score === "number" ? annotation.score : null;
}

export function recordCost(record: ExperimentRecord): number | null {
  const annotation = (record.annotations ?? []).find(
    (item) => String(item.name).toLowerCase() === "cost",
  );
  if (typeof annotation?.score === "number") return annotation.score;
  return typeof record.calculated_cost === "number" ? record.calculated_cost : null;
}

export function averageRecordCost(records: ExperimentRecord[]): number | null {
  const values = records.map(recordCost).filter((value): value is number => value != null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}
