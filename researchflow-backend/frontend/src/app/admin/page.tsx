"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface AdminStats {
  paper_states: Record<string, number>;
  total_papers: number;
  analysis_coverage: {
    l3_skim: number;
    l4_deep: number;
    delta_cards_published: number;
    coverage_pct: number;
  };
  enrichment: {
    with_abstract: number;
    with_doi: number;
    with_code: number;
    with_pdf: number;
  };
  review_queue: {
    pending: number;
    by_type: Record<string, number>;
  };
  candidates: {
    paradigms_pending: number;
    lineage_pending: number;
  };
  recent_7d: {
    imports: number;
    analyses: number;
  };
}

function StatCard({ label, value, sub, color = "blue" }: { label: string; value: number | string; sub?: string; color?: string }) {
  return (
    <div className={`bg-${color}-50 border border-${color}-200 rounded-lg p-4`}>
      <div className={`text-2xl font-bold text-${color}-700`}>{value}</div>
      <div className={`text-sm text-${color}-600`}>{label}</div>
      {sub && <div className="text-xs text-gray-400 mt-1">{sub}</div>}
    </div>
  );
}

function BarChart({ data, color = "blue" }: { data: Record<string, number>; color?: string }) {
  const max = Math.max(...Object.values(data), 1);
  return (
    <div className="space-y-1">
      {Object.entries(data).sort((a, b) => b[1] - a[1]).map(([key, val]) => (
        <div key={key} className="flex items-center gap-2 text-xs">
          <span className="w-32 text-right text-gray-500 truncate">{key}</span>
          <div className="flex-1 bg-gray-100 rounded h-4 overflow-hidden">
            <div
              className={`bg-${color}-500 h-full rounded`}
              style={{ width: `${(val / max) * 100}%` }}
            />
          </div>
          <span className="w-8 text-gray-600 font-mono">{val}</span>
        </div>
      ))}
    </div>
  );
}

export default function AdminPage() {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => { loadStats(); }, []);

  async function loadStats() {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/graph/admin-stats`);
      setStats(await res.json());
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  async function refreshViews() {
    setRefreshing(true);
    try {
      await fetch(`${API_BASE}/search/refresh-views`, { method: "POST" });
      await loadStats();
    } catch (e) { console.error(e); }
    setRefreshing(false);
  }

  if (loading) return <div className="p-8 text-gray-400">Loading admin stats...</div>;
  if (!stats) return <div className="p-8 text-red-500">Failed to load stats</div>;

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Admin Dashboard</h1>
        <button
          onClick={refreshViews}
          disabled={refreshing}
          className="text-sm bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 disabled:opacity-50"
        >
          {refreshing ? "Refreshing..." : "Refresh Views"}
        </button>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Papers" value={stats.total_papers} color="blue" />
        <StatCard label="L4 Analyzed" value={stats.analysis_coverage.l4_deep}
          sub={`${stats.analysis_coverage.coverage_pct}% coverage`} color="green" />
        <StatCard label="Review Pending" value={stats.review_queue.pending} color="yellow" />
        <StatCard label="7d Imports" value={stats.recent_7d.imports}
          sub={`${stats.recent_7d.analyses} analyses`} color="purple" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Paper State Distribution */}
        <div className="bg-white border rounded-lg p-4">
          <h2 className="font-semibold mb-3">Paper States</h2>
          <BarChart data={stats.paper_states} color="blue" />
        </div>

        {/* Enrichment Coverage */}
        <div className="bg-white border rounded-lg p-4">
          <h2 className="font-semibold mb-3">Enrichment Coverage</h2>
          <BarChart data={{
            "Abstract": stats.enrichment.with_abstract,
            "DOI": stats.enrichment.with_doi,
            "Code URL": stats.enrichment.with_code,
            "PDF": stats.enrichment.with_pdf,
          }} color="green" />
          <div className="mt-2 text-xs text-gray-400">
            out of {stats.total_papers} papers
          </div>
        </div>

        {/* Analysis Pipeline */}
        <div className="bg-white border rounded-lg p-4">
          <h2 className="font-semibold mb-3">Analysis Pipeline</h2>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span>L3 Skim</span>
              <span className="font-mono">{stats.analysis_coverage.l3_skim}</span>
            </div>
            <div className="w-full bg-gray-100 rounded h-3">
              <div className="bg-blue-400 h-full rounded" style={{
                width: `${(stats.analysis_coverage.l3_skim / (stats.total_papers || 1)) * 100}%`
              }} />
            </div>
            <div className="flex justify-between text-sm">
              <span>L4 Deep</span>
              <span className="font-mono">{stats.analysis_coverage.l4_deep}</span>
            </div>
            <div className="w-full bg-gray-100 rounded h-3">
              <div className="bg-green-500 h-full rounded" style={{
                width: `${(stats.analysis_coverage.l4_deep / (stats.total_papers || 1)) * 100}%`
              }} />
            </div>
            <div className="flex justify-between text-sm">
              <span>DeltaCards Published</span>
              <span className="font-mono">{stats.analysis_coverage.delta_cards_published}</span>
            </div>
            <div className="w-full bg-gray-100 rounded h-3">
              <div className="bg-purple-500 h-full rounded" style={{
                width: `${(stats.analysis_coverage.delta_cards_published / (stats.total_papers || 1)) * 100}%`
              }} />
            </div>
          </div>
        </div>

        {/* Review & Candidates */}
        <div className="bg-white border rounded-lg p-4">
          <h2 className="font-semibold mb-3">Review Queue & Candidates</h2>
          <div className="space-y-2">
            {Object.entries(stats.review_queue.by_type).map(([type, count]) => (
              <div key={type} className="flex justify-between text-sm">
                <span className="text-gray-600">{type}</span>
                <span className="font-mono bg-yellow-100 px-2 rounded">{count}</span>
              </div>
            ))}
            {Object.keys(stats.review_queue.by_type).length === 0 && (
              <div className="text-sm text-gray-400">No pending reviews</div>
            )}
            <hr className="my-2" />
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Paradigm candidates</span>
              <span className="font-mono">{stats.candidates.paradigms_pending}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Lineage candidates</span>
              <span className="font-mono">{stats.candidates.lineage_pending}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
