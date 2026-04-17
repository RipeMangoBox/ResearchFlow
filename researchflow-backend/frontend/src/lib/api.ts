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
};
