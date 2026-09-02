import { getMetricsCatalog } from "../api/metrics";
import type { MetricCatalogEntry } from "../api/types";

type RangeState = "ok" | "warn" | "unknown";
type ScoreDomain = NonNullable<MetricCatalogEntry["output_contract"]>["score_domain"];

const FALLBACK_DESC: Record<string, string> = {
  total_token: "Total prompt and completion tokens consumed by the selected run.",
};

function metricEntries(): MetricCatalogEntry[] {
  const out: MetricCatalogEntry[] = [];
  const cats = getMetricsCatalog().categories ?? {};
  for (const category of Object.values(cats)) {
    for (const group of Object.values(category.groups ?? {})) {
      out.push(...(group.metrics ?? []));
    }
  }
  return out;
}

export function getMetricCatalogEntry(name: string): MetricCatalogEntry | null {
  const key = String(name || "").toLowerCase();
  return metricEntries().find((m) => String(m.name).toLowerCase() === key) ?? null;
}

function fmtNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
}

function metricScaleText(entry: MetricCatalogEntry | null): string {
  const domain = entry?.output_contract?.score_domain;
  let scale = "";
  if (domain?.kind === "discrete_enum" && domain.allowed_values?.length) {
    scale = `${fmtNumber(Math.min(...domain.allowed_values))} to ${fmtNumber(Math.max(...domain.allowed_values))}`;
  } else if (domain?.kind === "continuous_range" || domain?.kind === "discrete_integer_range") {
    const minimum = domain.minimum ?? 0;
    scale = domain.maximum == null
      ? `${fmtNumber(minimum)} or more`
      : `${fmtNumber(minimum)} to ${fmtNumber(domain.maximum)}`;
  } else if (entry?.score_type === "binary" || entry?.score_type === "graded") {
    scale = "0 to 1";
  } else if (entry?.score_type === "magnitude") {
    scale = "a non-negative raw value";
  }
  const direction = entry?.higher_is_better === true
    ? "higher is better"
    : entry?.higher_is_better === false
      ? "lower is better"
      : "use it as a diagnostic value";
  return scale ? `How to read: ${scale}; ${direction}.` : `How to read: ${direction}.`;
}

function inDisplayedRange(domain: ScoreDomain, value: number): boolean | null {
  if (!domain) return null;
  if (domain.kind === "discrete_enum" && domain.allowed_values?.length) {
    return value >= Math.min(...domain.allowed_values) && value <= Math.max(...domain.allowed_values);
  }
  if (domain.kind === "continuous_range" || domain.kind === "discrete_integer_range") {
    const min = domain.minimum;
    const max = domain.maximum;
    return (min == null || value >= min) && (max == null || value <= max);
  }
  return null;
}

export function metricRangeState(name: string, value: number | null | undefined): RangeState {
  if (typeof value !== "number" || !Number.isFinite(value)) return "unknown";
  const key = String(name || "").toLowerCase();
  const entry = getMetricCatalogEntry(name);
  const contractState = inDisplayedRange(entry?.output_contract?.score_domain, value);
  if (contractState != null) return contractState ? "ok" : "warn";
  if (entry?.score_type === "binary" || entry?.score_type === "graded") {
    return value >= 0 && value <= 1 ? "ok" : "warn";
  }
  if (entry?.score_type === "magnitude" || key === "total_token") {
    return value >= 0 ? "ok" : "warn";
  }
  return "unknown";
}

export function metricRangeClass(name: string, value: number | null | undefined): string {
  const state = metricRangeState(name, value);
  return state === "ok" ? "range-ok" : state === "warn" ? "range-warn" : "range-unknown";
}

export function getMetricDescription(name: string, value?: number | null): string {
  const key = String(name || "").toLowerCase();
  const entry = getMetricCatalogEntry(name);
  const meaning = entry?.plain_language || entry?.desc || FALLBACK_DESC[key] || "Evaluator metric: " + name + ".";
  const rangeState = metricRangeState(name, value);
  return [
    "- " + meaning,
    "- " + metricScaleText(entry),
    rangeState === "warn" ? "- This value is outside the expected range." : "",
  ]
    .filter(Boolean)
    .join("\n");
}
