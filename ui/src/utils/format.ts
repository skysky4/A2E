export function esc(s: unknown): string {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] as string);
}

export function pretty(v: unknown): string {
  try {
    return typeof v === "string" ? v : JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

function decodeHtmlEntities(value: string): string {
  return value
    .replace(/&quot;/g, '"')
    .replace(/&#34;/g, '"')
    .replace(/&apos;|&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

function parseNestedJson(value: unknown, depth = 0): unknown {
  if (depth > 4) return value;
  if (typeof value === "string") {
    const decoded = decodeHtmlEntities(value);
    const trimmed = decoded.trim();
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
      try {
        return parseNestedJson(JSON.parse(trimmed), depth + 1);
      } catch {
        return decoded;
      }
    }
    return decoded;
  }
  if (Array.isArray(value)) return value.map((item) => parseNestedJson(item, depth + 1));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, parseNestedJson(item, depth + 1)]));
  }
  return value;
}

export function prettyReadable(value: unknown): string {
  return pretty(parseNestedJson(value));
}

export function avgNumber(xs: (number | undefined | null)[]): number | null {
  const nums = xs.filter((x): x is number => typeof x === "number");
  return nums.length ? nums.reduce((s, x) => s + x, 0) / nums.length : null;
}
