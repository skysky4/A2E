import type { ExperimentRecord } from "../api/types";
import pricing from "../data/modelPricing.json";

interface ModelPrice {
  input_usd_per_1m_tokens: number | null;
  output_usd_per_1m_tokens: number | null;
}

function normalizeModelName(value: string): string {
  return value.trim().toLowerCase();
}

function modelPrice(modelName: string | undefined): ModelPrice | null {
  if (!modelName) return null;
  const normalized = normalizeModelName(modelName);
  for (const [name, price] of Object.entries(pricing.models)) {
    const normalizedName = normalizeModelName(name);
    if (normalized === normalizedName || normalized.endsWith(`/${normalizedName}`)) {
      return price;
    }
  }
  return null;
}

function hasCostAnnotation(record: ExperimentRecord): boolean {
  return (record.annotations ?? []).some(
    (annotation) => String(annotation.name).toLowerCase() === "cost" && typeof annotation.score === "number",
  );
}

export function applyModelPricing(
  records: ExperimentRecord[],
  modelName: string | undefined,
): ExperimentRecord[] {
  const price = modelPrice(modelName);
  if (!price) return records;
  const tokenUnit = pricing.token_unit;
  return records.map((record) => {
    if (hasCostAnnotation(record)) return record;
    let cost = 0;
    let priced = false;
    if (price.input_usd_per_1m_tokens != null && typeof record.prompt_token_count === "number") {
      cost += (record.prompt_token_count * price.input_usd_per_1m_tokens) / tokenUnit;
      priced = true;
    }
    if (price.output_usd_per_1m_tokens != null && typeof record.completion_token_count === "number") {
      cost += (record.completion_token_count * price.output_usd_per_1m_tokens) / tokenUnit;
      priced = true;
    }
    return priced ? { ...record, calculated_cost: cost } : record;
  });
}
