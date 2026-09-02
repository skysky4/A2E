import type { SpanNode } from "../api/types";
import { pretty } from "./format";

const KIND: Record<string, { c: string }> = {
  LLM: { c: "#7c5cff" },
  PROMPT: { c: "#9b5de5" },
  CHAIN: { c: "#1f7ae0" },
  AGENT: { c: "#e83e8c" },
  TOOL: { c: "#c98f00" },
  RETRIEVER: { c: "#0f9fbc" },
  EMBEDDING: { c: "#7b61ff" },
  RERANKER: { c: "#45a834" },
  EVALUATOR: { c: "#7b61ff" },
  GUARDRAIL: { c: "#e63946" },
  UNKNOWN: { c: "#7c8794" },
};

export const INDENT = 34;

export function spanId(span: SpanNode, index: number): string {
  return span.span_id || `span-${index}`;
}

export function semanticSpanKind(span: SpanNode): string {
  const recordedKind = String(span.span_kind || "UNKNOWN").trim().toUpperCase() || "UNKNOWN";
  if (recordedKind !== "UNKNOWN") return recordedKind;

  const name = String(span.name || "").trim();
  if (/^execute_tool(?:\b|[._])/i.test(name)) return "TOOL";
  if (/^campaign\.trial\./i.test(name)) return "CHAIN";
  if (/^(?:create|invoke)_agent(?:\b|[._])/i.test(name)) return "AGENT";
  if (/^call_llm(?:\b|[._])/i.test(name)) return "LLM";
  return recordedKind;
}

function humanizeIdentifier(value: string): string {
  const words = value
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return words ? `${words.charAt(0).toUpperCase()}${words.slice(1)}` : "Workflow step";
}

const CHAIN_NAME_OVERRIDES: Record<string, string> = {
  kickoff: "Start workflow",
  generate_plan: "Generate plan",
  check_todos_available: "Check available tasks",
  initialize_reasoning: "Initialize reasoning",
  check_max_iterations: "Check iteration limit",
  call_llm_native_tools: "Call LLM",
  execute_native_tool: "Execute tool",
  check_native_todo_completion: "Check task completion",
  increment_and_continue: "Continue to next iteration",
  continue_iteration: "Continue iteration",
  aggregate_tool_results: "Aggregate tool results",
  call_tool: "Call tool",
  setup_agent: "Set up agent",
  run_agent_step: "Run agent step",
  parse_agent_output: "Parse agent output",
};

function chainStepName(value: string): string {
  return CHAIN_NAME_OVERRIDES[value] || humanizeIdentifier(value);
}

export function displaySpanName(span: SpanNode): string {
  const name = String(span.name || "").trim();
  if (!name) return "Span";

  const flow = name.match(/^Flow_[0-9a-f-]+(?:\.(.+))?$/i);
  if (flow) return flow[1] ? chainStepName(flow[1]) : "Agent workflow";

  const workflowAgent = name.match(/^BaseWorkflowAgent\.(.+)$/i);
  if (workflowAgent) return chainStepName(workflowAgent[1]);
  if (/^FunctionAgent\.run$/i.test(name)) return "Run agent";
  if (/^campaign\.trial\./i.test(name)) return "Evaluation trial";
  if (/^Task:\s*task_fn$/i.test(name)) return "Task run";
  if (/^_branch_after_router$/i.test(name)) return "Route result";
  if (/^router$/i.test(name)) return "Route request";
  if (/^executor$/i.test(name)) return "Execute step";
  if (/^turn$/i.test(name)) return "Agent turn";
  if (/^call_llm$/i.test(name)) return "Call LLM";

  const tool = name.match(/^execute_tool(?:\s+(.+))?$/i);
  if (tool) return tool[1] ? `Execute tool: ${humanizeIdentifier(tool[1])}` : "Execute tool";
  const agent = name.match(/^(create|invoke)_agent(?:\s+(.+))?$/i);
  if (agent) {
    const action = agent[1].toLowerCase() === "create" ? "Create agent" : "Invoke agent";
    return agent[2] ? `${action}: ${humanizeIdentifier(agent[2])}` : action;
  }

  return name;
}

function isInternalSpan(span: SpanNode): boolean {
  const name = String(span.name || "");
  const kind = String(span.span_kind || "").toUpperCase();
  return (
    /^_branch_after_/i.test(name) ||
    /^Task:\s*task_fn$/i.test(name) ||
    (kind === "AGENT" && /^agent\./i.test(name))
  );
}

function nearestVisibleContainer(span: SpanNode, visible: SpanNode[]): string | null {
  const selfId = span.span_id;
  const s0 = +new Date(span.start_time);
  const s1 = +new Date(span.end_time);
  let best: string | null = null;
  let bestDur = Infinity;
  visible.forEach((candidate, i) => {
    const id = spanId(candidate, i);
    if (id === selfId) return;
    const c0 = +new Date(candidate.start_time);
    const c1 = +new Date(candidate.end_time);
    if (c0 <= s0 && c1 >= s1) {
      const dur = c1 - c0;
      if (dur < bestDur) {
        best = id;
        bestDur = dur;
      }
    }
  });
  return best;
}

export function normalizeVisibleSpans(
  spans: SpanNode[],
  { filterInternal = true }: { filterInternal?: boolean } = {},
): SpanNode[] {
  const original = new Map<string, SpanNode>();
  spans.forEach((s, i) => original.set(spanId(s, i), s));
  const visible = filterInternal ? spans.filter((s) => !isInternalSpan(s)) : spans;
  const visibleIds = new Set(visible.map((s, i) => spanId(s, i)));
  return visible.map((s) => {
    let parent = s.parent_id ?? null;
    while (parent && !visibleIds.has(parent)) {
      parent = original.get(parent)?.parent_id ?? null;
    }
    if (!parent && s.parent_id) {
      parent = nearestVisibleContainer(s, visible);
    }
    if (parent === s.parent_id) return s;
    return { ...s, parent_id: parent };
  });
}

function childTime(parent: SpanNode, index: number, total: number): { start_time: string; end_time: string } {
  const start = +new Date(parent.start_time);
  const end = +new Date(parent.end_time);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return { start_time: parent.start_time, end_time: parent.end_time };
  }
  const step = Math.max(1, Math.floor((end - start) / Math.max(1, total)));
  const s = start + step * index;
  const e = Math.min(end, s + Math.max(1, Math.floor(step * 0.7)));
  return { start_time: new Date(s).toISOString(), end_time: new Date(e).toISOString() };
}

interface TraceMessage {
  role?: string;
  content?: unknown;
  tool_calls?: unknown;
  tool_call_id?: unknown;
  name?: string;
  raw?: unknown;
}

function messageText(message: TraceMessage): string {
  if (typeof message.content === "string") return message.content;
  if (message.content != null) return pretty(message.content);
  if (message.tool_calls != null) return pretty(message.tool_calls);
  return pretty(message.raw ?? message);
}

function toolIntentFromAssistant(content: string): { name: string; command: string } | null {
  const match = content.match(/(?:^|\n)(bash|str_replace_editor)\s*\n(?:\1\s*\n)?([\s\S]+)$/i);
  if (!match) return null;
  return { name: match[1], command: match[2].trim() };
}

export function expandTraceConversationSpans(spans: SpanNode[]): SpanNode[] {
  const hasToolSpans = spans.some((s) => semanticSpanKind(s) === "TOOL");
  const expanded: SpanNode[] = [...spans];
  const seenSyntheticIds = new Set(expanded.map((s, i) => spanId(s, i)));
  for (const span of spans) {
    const kind = semanticSpanKind(span);
    const attrs = span.attributes ?? {};
    if (kind !== "LLM") continue;
    const parent = span.span_id;
    const inputMessages = collectMessages(attrs, "llm.input_messages");
    const outputMessages = collectMessages(attrs, "llm.output_messages");
    const syntheticCount = (inputMessages.length ? 1 : 0) + outputMessages.length;
    if (inputMessages.length) {
      const inputId = `${parent}:input-messages`;
      if (!seenSyntheticIds.has(inputId)) {
        seenSyntheticIds.add(inputId);
        expanded.push({
          span_id: inputId,
          parent_id: parent,
          name: `LLM Input · ${inputMessages.length} ${inputMessages.length === 1 ? "message" : "messages"}`,
          span_kind: "PROMPT",
          ...childTime(span, 1, syntheticCount + 2),
          attributes: { "llm.input_messages": inputMessages },
          synthetic: true,
        });
      }
    }
    outputMessages.forEach((message, index) => {
      const role = message.role || "message";
      const text = messageText(message);
      const timeIndex = index + (inputMessages.length ? 2 : 1);
      const time = childTime(span, timeIndex, syntheticCount + 2);
      const messageId = `${parent}:output-message:${index}`;
      if (seenSyntheticIds.has(messageId)) return;
      seenSyntheticIds.add(messageId);
      expanded.push({
        span_id: messageId,
        parent_id: parent,
        name: `Output: ${role}`,
        span_kind: "LLM",
        ...time,
        attributes: { output: text, role },
        synthetic: true,
      });
      if (!hasToolSpans) {
        const toolIntent = toolIntentFromAssistant(text);
        if (toolIntent) {
          const toolTime = childTime(span, timeIndex + 1, syntheticCount + 2);
          const toolId = `${parent}:tool-intent:${index}`;
          if (seenSyntheticIds.has(toolId)) return;
          seenSyntheticIds.add(toolId);
          expanded.push({
            span_id: toolId,
            parent_id: messageId,
            name: `Tool intent: ${toolIntent.name}`,
            span_kind: "TOOL",
            ...toolTime,
            attributes: { input: toolIntent.command },
            synthetic: true,
          });
        }
      }
    });
  }
  return expanded;
}

export const expandCompactTraceSpans = expandTraceConversationSpans;

export function fmtMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function kindColor(kind?: string): string {
  const k = (kind || "UNKNOWN").toUpperCase();
  return (KIND[k] ?? KIND.UNKNOWN).c;
}

export function kindSummary(spans: SpanNode[]): string {
  const counts: Record<string, number> = {};
  spans.forEach((s) => {
    const k = semanticSpanKind(s);
    counts[k] = (counts[k] || 0) + 1;
  });
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([k, n]) => `${n} ${k.toLowerCase()}`)
    .join(" · ");
}

export function attrValue(attrs: Record<string, unknown>, path: string): unknown {
  if (attrs[path] != null) return attrs[path];
  const parts = path.split(".");
  let cur: unknown = attrs;
  for (const part of parts) {
    if (!cur || typeof cur !== "object") return null;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur;
}

export function collectMessages(
  attrs: Record<string, unknown>,
  prefix: string,
): TraceMessage[] {
  const nested = attrValue(attrs, prefix);
  if (Array.isArray(nested)) {
    return nested.map((m: Record<string, unknown>) => ({
      role: (m?.message as { role?: string })?.role ?? (m?.role as string),
      content: (m?.message as { content?: unknown })?.content ?? m?.content,
      tool_calls: (m?.message as { tool_calls?: unknown })?.tool_calls ?? m?.tool_calls,
      tool_call_id: (m?.message as { tool_call_id?: unknown })?.tool_call_id ?? m?.tool_call_id,
      name: (m?.message as { name?: string })?.name ?? (m?.name as string),
      raw: m,
    }));
  }
  if (Array.isArray(attrs[prefix])) {
    return (attrs[prefix] as Record<string, unknown>[]).map((m) => ({
      role: (m?.message as { role?: string })?.role ?? (m?.role as string),
      content: (m?.message as { content?: unknown })?.content ?? m?.content,
      tool_calls: (m?.message as { tool_calls?: unknown })?.tool_calls ?? m?.tool_calls,
      tool_call_id: (m?.message as { tool_call_id?: unknown })?.tool_call_id ?? m?.tool_call_id,
      name: (m?.message as { name?: string })?.name ?? (m?.name as string),
      raw: m,
    }));
  }
  const out: (TraceMessage | undefined)[] = [];
  Object.keys(attrs).forEach((key) => {
    const m = key.match(
      new RegExp(`^${prefix.replace(".", "\\.")}\\.(\\d+)\\.message\\.(role|content|tool_calls|tool_call_id|name)$`),
    );
    if (!m) return;
    const idx = +m[1];
    out[idx] ||= {};
    if (m[2] === "role") out[idx]!.role = String(attrs[key]);
    else if (m[2] === "content") out[idx]!.content = attrs[key];
    else if (m[2] === "tool_calls") out[idx]!.tool_calls = attrs[key];
    else if (m[2] === "tool_call_id") out[idx]!.tool_call_id = attrs[key];
    else out[idx]!.name = String(attrs[key]);
  });
  return out.filter(Boolean) as TraceMessage[];
}

export function tokenTotal(attrs: Record<string, unknown> = {}): number | null {
  const total = attrValue(attrs, "llm.token_count.total");
  if (typeof total === "number") return total;
  const prompt = (attrValue(attrs, "llm.token_count.prompt") as number) || 0;
  const completion = (attrValue(attrs, "llm.token_count.completion") as number) || 0;
  return prompt || completion ? prompt + completion : null;
}

export function trunc(s: string, n = 1600): string {
  return s.length > n ? `${s.slice(0, n)} …(truncated)` : s;
}

export interface TreeNode {
  span: SpanNode;
  children: TreeNode[];
  rangeStart: number;
  rangeEnd: number;
}

export function buildTree(spans: SpanNode[]): { roots: TreeNode[]; t0: number; total: number } {
  const byId = new Map<string, TreeNode>();
  spans.forEach((s, i) => {
    const start = toTime(s.start_time);
    const end = Math.max(start, toTime(s.end_time));
    byId.set(spanId(s, i), { span: s, children: [], rangeStart: start, rangeEnd: end });
  });
  const roots: TreeNode[] = [];
  spans.forEach((s, i) => {
    const node = byId.get(spanId(s, i))!;
    const parent = s.parent_id ? byId.get(s.parent_id) : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node);
  });
  const start = (n: TreeNode) => +new Date(n.span.start_time);
  const sortRec = (n: TreeNode) => {
    n.children.sort((a, b) => start(a) - start(b));
    n.children.forEach(sortRec);
  };
  roots.sort((a, b) => start(a) - start(b));
  roots.forEach(sortRec);

  const rangeRec = (n: TreeNode): { start: number; end: number } => {
    let rangeStart = n.rangeStart;
    let rangeEnd = n.rangeEnd;
    for (const child of n.children) {
      const childRange = rangeRec(child);
      rangeStart = Math.min(rangeStart, childRange.start);
      rangeEnd = Math.max(rangeEnd, childRange.end);
    }
    n.rangeStart = rangeStart;
    n.rangeEnd = Math.max(rangeStart, rangeEnd);
    return { start: n.rangeStart, end: n.rangeEnd };
  };
  roots.forEach(rangeRec);

  const t0 = Math.min(...roots.map((n) => n.rangeStart));
  const t1 = Math.max(...roots.map((n) => n.rangeEnd));
  return { roots, t0, total: Math.max(1, t1 - t0) };
}

export function isTraceRootNode(node: TreeNode): boolean {
  return /^LangGraph$/i.test(String(node.span.name || ""));
}

export function toTime(value: string): number {
  const parsed = +new Date(value);
  return Number.isFinite(parsed) ? parsed : 0;
}
