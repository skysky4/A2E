import metricsCatalog from "../../../eval/metrics_catalog.json";
import type { MetricsCatalog } from "./types";

export function getMetricsCatalog(): MetricsCatalog {
  return metricsCatalog as unknown as MetricsCatalog;
}

export function catalogGroupMetrics(group: string): string[] {
  const cats = getMetricsCatalog().categories ?? {};
  for (const top of Object.values(cats)) {
    const g = top?.groups?.[group];
    if (g?.metrics) return g.metrics.map((m) => m.name);
  }
  return [];
}

export function catalogMetricGroups(): Array<{ id: string; label: string; metrics: string[] }> {
  const groups: Array<{ id: string; label: string; metrics: string[] }> = [];
  const cats = getMetricsCatalog().categories ?? {};
  for (const top of Object.values(cats)) {
    for (const [id, group] of Object.entries(top.groups ?? {})) {
      groups.push({
        id,
        label: group.label ?? id,
        metrics: (group.metrics ?? []).map((metric) => metric.name),
      });
    }
  }
  return groups;
}
