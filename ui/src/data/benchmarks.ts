export const CATS = ["Coding", "Reasoning", "Research & Work", "Tool Use"] as const;
export const CAPS = ["Skill", "Memory", "Tool"] as const;

export type BenchDiff = "found" | "med" | "hard" | "front";

export interface Benchmark {
  name: string;
  cat?: number;
  year: string;
  date?: string;
  diff: BenchDiff;
  dim?: number;
  key?: string;
  aliases?: string[];
  experimentIds?: string[];
}

export const BENCHMARKS: Benchmark[] = [
  { name: "HumanEval", cat: 0, year: "2021", diff: "found", key: "humaneval", aliases: ["human-eval"] },
  { name: "SWE-bench Lite", cat: 0, year: "2024", diff: "hard", dim: 2, key: "swe-bench-lite" },
  { name: "SWE-bench Verified", cat: 0, year: "2024", diff: "hard", dim: 2, key: "swe-bench-verified" },
  { name: "SWE-bench Pro", cat: 0, year: "2025", diff: "front", dim: 2, key: "swe-bench-pro" },
  { name: "AGIEval", cat: 1, year: "2023", diff: "med", key: "agieval", aliases: ["agi-eval"] },
  { name: "ARC-Challenge", cat: 1, year: "2018", diff: "med", key: "arc-challenge", aliases: ["arc_challenge"] },
  { name: "BBH", cat: 1, year: "2022", diff: "med", key: "bbh", aliases: ["big-bench-hard"] },
  { name: "CommonsenseQA", cat: 1, year: "2018", diff: "med", key: "commonsenseqa", aliases: ["commonsense-qa"] },
  { name: "GSM8K", cat: 1, year: "2021", diff: "med", key: "gsm8k" },
  { name: "HellaSwag", cat: 1, year: "2019", diff: "med", key: "hellaswag", aliases: ["hella-swag"] },
  { name: "MATH", cat: 1, year: "2021", diff: "hard", key: "math" },
  { name: "MMLU", cat: 1, year: "2020", diff: "med", key: "mmlu" },
  { name: "MMLU-Pro", cat: 1, year: "2024", diff: "hard", key: "mmlu-pro", aliases: ["mmlupro"] },
  { name: "OpenBookQA", cat: 1, year: "2018", diff: "med", key: "openbookqa", aliases: ["openbook-qa"] },
  { name: "TruthfulQA", cat: 1, year: "2021", diff: "med", key: "truthfulqa", aliases: ["truthful-qa"] },
  { name: "GPQA", cat: 1, year: "2023", diff: "hard", key: "gpqa" },
  { name: "DeepSearchQA", cat: 2, year: "2025", diff: "front", dim: 2, key: "deepsearchqa", aliases: ["deep-search-qa"] },
  { name: "GDPval", cat: 2, year: "2025", diff: "front", dim: 2, key: "gdpval", aliases: ["gdp-val"] },
  { name: "Traject-Bench", cat: 3, year: "2025", diff: "hard", dim: 2, key: "traject-bench", aliases: ["trajectbench"] },
  { name: "τ-bench", cat: 3, year: "2024", diff: "hard", dim: 2, key: "tau-bench", aliases: ["taubench"] },
  { name: "τ²-bench", cat: 3, year: "2025", diff: "front", dim: 2, key: "tau2-bench", aliases: ["tau2", "tau2bench"] },
  { name: "τ³-bench", cat: 3, year: "2026", diff: "front", dim: 2, key: "tau3-bench", aliases: ["tau3", "tau3bench"] },
  { name: "Terminal-Bench 2.1", cat: 0, year: "2026", diff: "front", dim: 2, key: "terminal-bench-2.1", aliases: ["terminalbench21", "terminal-bench-2-1", "tb21"] },
];

export function normKey(s: string): string {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

export function benchKey(b: Benchmark): string {
  return b.key || b.name;
}

import type { ExperimentSummary } from "../api/types";

const BENCHMARK_META_KEYS = [
  "benchmark",
  "benchmark_name",
  "bench",
  "dataset",
  "dataset_name",
  "task",
  "task_name",
  "suite",
] as const;

function experimentBenchmarkLabels(exp: ExperimentSummary): string[] {
  const meta = exp.metadata ?? {};
  return [
    ...BENCHMARK_META_KEYS.map((key) => meta[key]),
    exp.dataset_name,
    exp.name,
    exp.project_name,
  ]
    .filter((value): value is string | number => typeof value === "string" || typeof value === "number")
    .map((value) => String(value).trim())
    .filter(Boolean);
}

function benchmarkAliases(benchmark: Benchmark): string[] {
  return [benchKey(benchmark), benchmark.name, ...(benchmark.aliases ?? [])]
    .map(normKey)
    .filter(Boolean)
    .filter((alias, index, aliases) => aliases.indexOf(alias) === index)
    .sort((a, b) => b.length - a.length);
}

const BENCHMARK_MATCHERS = BENCHMARKS.flatMap((benchmark) =>
  benchmarkAliases(benchmark).map((alias) => ({ benchmark, alias })),
).sort((a, b) => b.alias.length - a.alias.length);

function staticBenchmarkForExperiment(exp: ExperimentSummary): Benchmark | undefined {
  for (const label of experimentBenchmarkLabels(exp)) {
    const normalized = normKey(label);
    if (!normalized) continue;
    const exact = BENCHMARK_MATCHERS.find(({ alias }) => normalized === alias);
    if (exact) return exact.benchmark;
    const contained = BENCHMARK_MATCHERS.find(({ alias }) => normalized.includes(alias));
    if (contained) return contained.benchmark;
  }
  return undefined;
}

export function benchmarksFromExperiments(experiments: ExperimentSummary[]): Benchmark[] {
  const experimentIdsByBenchmark = new Map<string, string[]>();
  for (const exp of experiments) {
    const benchmark = staticBenchmarkForExperiment(exp);
    if (!benchmark) continue;
    const key = normKey(benchKey(benchmark));
    const ids = experimentIdsByBenchmark.get(key) ?? [];
    if (!ids.includes(exp.id)) ids.push(exp.id);
    experimentIdsByBenchmark.set(key, ids);
  }
  return BENCHMARKS.map((benchmark) => ({
    ...benchmark,
    experimentIds: experimentIdsByBenchmark.get(normKey(benchKey(benchmark))) ?? [],
  }));
}

export function benchExperiments(b: Benchmark, experiments: ExperimentSummary[]): ExperimentSummary[] {
  if (b.experimentIds) {
    const ids = new Set(b.experimentIds);
    return experiments.filter((e) => ids.has(e.id));
  }
  const key = normKey(benchKey(b));
  if (!key) return [];
  return experiments.filter((e) => {
    const metaAgent = String(e.metadata?.agent ?? "");
    const haystack = normKey(`${e.name || ""} ${e.dataset_name || ""} ${e.project_name || ""} ${metaAgent}`);
    return haystack.includes(key);
  });
}
