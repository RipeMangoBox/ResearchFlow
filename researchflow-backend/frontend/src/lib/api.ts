const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  if (res.status === 204) return {} as T;
  return res.json();
}

// ── Types ──────────────────────────────────────────────────────

export interface PaperBrief {
  id: string;
  title: string;
  venue: string | null;
  year: number | null;
  category: string;
  state: string;
  importance: string | null;
  tier: string | null;
  tags: string[];
  core_operator: string | null;
  keep_score: number | null;
  structurality_score: number | null;
  is_ephemeral: boolean;
}

export interface PaperDetail extends PaperBrief {
  title_sanitized: string;
  abstract: string | null;
  paper_link: string | null;
  project_link: string | null;
  primary_logic: string | null;
  claims: string[] | null;
  open_code: boolean;
  open_data: boolean;
  latest_analysis: AnalysisBrief | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface AnalysisBrief {
  id: string;
  level: string;
  model_provider: string | null;
  model_name: string | null;
  confidence: number | null;
  problem_summary: string | null;
  method_summary: string | null;
  core_intuition: string | null;
  changed_slots: string[] | null;
  is_plugin_patch: boolean | null;
  worth_deep_read: boolean | null;
}

export interface PaperListResponse {
  items: PaperBrief[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

export interface SearchResult {
  paper_id: string;
  title: string;
  venue: string | null;
  year: number | null;
  category: string;
  text_score: number;
  vector_score: number;
  combined_score: number;
  core_operator: string;
  keep_score: number | null;
  structurality_score: number | null;
  tags: string[];
}

export interface ReadingPlanEntry {
  paper_id: string;
  title: string;
  venue: string | null;
  year: number | null;
  reading_depth: string;
  tier_reason: string;
  keep_score: number | null;
  structurality_score: number | null;
  core_operator: string;
}

export interface DigestInfo {
  id: string;
  period_start: string;
  period_end: string;
  content: string;
}

export interface DeltaCard {
  id: string;
  paper_id: string;
  analysis_id: string | null;
  frame_id: string | null;
  baseline_paradigm: string | null;
  primary_bottleneck_id: string | null;
  changed_slot_ids: string[] | null;
  unchanged_slot_ids: string[] | null;
  mechanism_family_ids: string[] | null;
  delta_statement: string;
  key_ideas_ranked: Record<string, unknown>[] | null;
  structurality_score: number | null;
  extensionability_score: number | null;
  transferability_score: number | null;
  assumptions: string[] | null;
  failure_modes: string[] | null;
  evaluation_context: string | null;
  evidence_refs: string[] | null;
  extraction_confidence: number | null;
  linkage_confidence: number | null;
  evidence_confidence: number | null;
  status: string;
  model_provider: string | null;
  model_name: string | null;
  prompt_version: string | null;
  schema_version: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface GraphAssertion {
  id: string;
  from_node_id: string;
  to_node_id: string;
  edge_type: string;
  assertion_source: string;
  confidence: number | null;
  status: string;
  reviewed_by: string | null;
  reviewed_at: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string | null;
}

export interface ReviewTask {
  id: string;
  target_type: string;
  target_id: string;
  task_type: string;
  status: string;
  priority: number;
  assigned_to: string | null;
  notes: string | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface Alias {
  id: string;
  entity_type: string;
  entity_id: string;
  alias: string;
  source: string;
  confidence: number | null;
}

// ── API calls ──────────────────────────────────────────────────

export const api = {
  // Papers
  listPapers: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<PaperListResponse>(`/papers${qs}`);
  },
  getPaper: (id: string) => request<PaperDetail>(`/papers/${id}`),
  triageAll: () => request<{ scored: number }>("/papers/triage-all", { method: "POST" }),
  enrichPapers: (limit = 10) =>
    request<{ processed: number }>(`/papers/enrich?limit=${limit}`, { method: "POST" }),

  // Import
  importLinks: (items: { url: string; title?: string; category?: string }[], category = "Uncategorized") =>
    request<{ total: number; created: number; duplicates: number; items: any[] }>("/import/links", {
      method: "POST",
      body: JSON.stringify({ items, default_category: category }),
    }),

  // Search
  hybridSearch: (query: string, opts?: { category?: string; limit?: number }) =>
    request<{ query: string; total: number; results: SearchResult[] }>("/search/hybrid", {
      method: "POST",
      body: JSON.stringify({ query, ...opts }),
    }),

  // Reports
  generateReport: (paperIds: string[], type = "briefing", topic?: string) =>
    request<{ report_type: string; content: string; paper_count: number }>("/reports/generate", {
      method: "POST",
      body: JSON.stringify({ paper_ids: paperIds, report_type: type, topic }),
    }),

  // Reading plan
  readingPlan: (category?: string, maxPapers = 15) => {
    const params = new URLSearchParams();
    if (category) params.set("category", category);
    params.set("max_papers", String(maxPapers));
    return request<{
      category: string | null;
      tiers: Record<string, ReadingPlanEntry[]>;
      total_papers: number;
      reading_time_estimate: string;
      reading_order: string;
    }>(`/reading-plan?${params}`, { method: "POST" });
  },

  // Digests
  generateDigest: (periodType: string) =>
    request<{ period_type: string; content: string; period_start: string; period_end: string }>(
      `/digests/generate?period_type=${periodType}`,
      { method: "POST" },
    ),
  latestDigests: () => request<Record<string, DigestInfo | null>>("/digests/latest"),

  // Analyses
  skimPaper: (id: string) => request<any>(`/analyses/${id}/skim`, { method: "POST" }),
  deepAnalyze: (id: string) => request<any>(`/analyses/${id}/deep`, { method: "POST" }),

  // Embeddings
  generateEmbeddings: (limit = 50) =>
    request<{ embedded: number }>(`/embeddings/generate?limit=${limit}`, { method: "POST" }),

  // Health
  health: () => request<{ status: string }>("/health"),

  // Ideas / Delta Cards
  searchIdeas: (query: string, options?: { category?: string; limit?: number; min_score?: number }) =>
    request<{ query: string; total: number; results: DeltaCard[] }>("/ideas/search", {
      method: "POST",
      body: JSON.stringify({ query, ...options }),
    }),

  // Graph
  getGraphStats: () =>
    request<{ node_count: number; assertion_count: number; edge_type_distribution: Record<string, number> }>(
      "/graph/stats",
    ),
  getAssertionDetail: (assertionId: string) =>
    request<GraphAssertion>(`/graph/assertions/${assertionId}`),

  // Review queue
  getReviewQueue: (status?: string, targetType?: string) => {
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    if (targetType) params.set("target_type", targetType);
    const qs = params.toString();
    return request<{ items: ReviewTask[]; total: number }>(`/review/tasks${qs ? `?${qs}` : ""}`);
  },
  submitReviewDecision: (taskId: string, decision: string, reviewer: string) =>
    request<ReviewTask>(`/review/tasks/${taskId}/decide`, {
      method: "POST",
      body: JSON.stringify({ decision, reviewer }),
    }),

  // Aliases
  listAliases: (entityType?: string, entityId?: string) => {
    const params = new URLSearchParams();
    if (entityType) params.set("entity_type", entityType);
    if (entityId) params.set("entity_id", entityId);
    const qs = params.toString();
    return request<{ items: Alias[]; total: number }>(`/aliases${qs ? `?${qs}` : ""}`);
  },
};
