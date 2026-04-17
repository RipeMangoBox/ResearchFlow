"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api, type PaperDetail } from "@/lib/api";

export default function PaperDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);

  useEffect(() => {
    api.getPaper(id).then(setPaper).catch(console.error).finally(() => setLoading(false));
  }, [id]);

  async function runSkim() {
    setAnalyzing(true);
    try {
      await api.skimPaper(id);
      const updated = await api.getPaper(id);
      setPaper(updated);
    } catch (e) { console.error(e); }
    setAnalyzing(false);
  }

  async function runDeep() {
    setAnalyzing(true);
    try {
      await api.deepAnalyze(id);
      const updated = await api.getPaper(id);
      setPaper(updated);
    } catch (e) { console.error(e); }
    setAnalyzing(false);
  }

  if (loading) return <div className="text-center py-20 text-gray-400">Loading...</div>;
  if (!paper) return <div className="text-center py-20 text-red-400">Paper not found</div>;

  const a = paper.latest_analysis;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-900">{paper.title}</h1>
        <div className="flex gap-3 mt-2 text-sm text-gray-500">
          <span>{paper.venue} {paper.year}</span>
          <span className="px-2 py-0.5 bg-gray-100 rounded text-xs">{paper.state}</span>
          {paper.importance && <span className="px-2 py-0.5 bg-orange-100 text-orange-700 rounded text-xs">{paper.importance}</span>}
          {paper.tier && <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">{paper.tier}</span>}
        </div>
      </div>

      {/* Scores */}
      <div className="grid grid-cols-4 gap-3">
        <ScoreBox label="Keep" value={paper.keep_score} />
        <ScoreBox label="Structurality" value={paper.structurality_score} />
        <ScoreBox label="Open Code" value={paper.open_code ? "Yes" : "No"} />
        <ScoreBox label="Ephemeral" value={paper.is_ephemeral ? "Yes" : "No"} />
      </div>

      {/* Links */}
      <div className="flex gap-4 text-sm">
        {paper.paper_link && <a href={paper.paper_link} target="_blank" className="text-blue-600 hover:underline">Paper Link</a>}
        {paper.project_link && <a href={paper.project_link} target="_blank" className="text-blue-600 hover:underline">Project Page</a>}
      </div>

      {/* Core Operator */}
      {paper.core_operator && (
        <Section title="Core Operator">
          <p className="text-sm text-gray-700">{paper.core_operator}</p>
        </Section>
      )}

      {/* Primary Logic */}
      {paper.primary_logic && (
        <Section title="Primary Logic">
          <pre className="text-sm text-gray-600 whitespace-pre-wrap bg-gray-50 p-3 rounded">{paper.primary_logic}</pre>
        </Section>
      )}

      {/* Abstract */}
      {paper.abstract && (
        <Section title="Abstract">
          <p className="text-sm text-gray-700">{paper.abstract}</p>
        </Section>
      )}

      {/* Tags */}
      {paper.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {paper.tags.map((t) => <span key={t} className="px-2 py-0.5 bg-gray-100 rounded text-xs text-gray-600">{t}</span>)}
        </div>
      )}

      {/* Analysis */}
      <div className="border-t pt-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-800">Analysis</h2>
          <div className="flex gap-2">
            <button onClick={runSkim} disabled={analyzing} className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50">
              {analyzing ? "..." : "L3 Skim"}
            </button>
            <button onClick={runDeep} disabled={analyzing} className="px-3 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-700 disabled:opacity-50">
              {analyzing ? "..." : "L4 Deep"}
            </button>
          </div>
        </div>

        {a ? (
          <div className="space-y-4">
            <div className="flex gap-3 text-xs text-gray-400">
              <span>Level: {a.level}</span>
              <span>Model: {a.model_provider}/{a.model_name}</span>
              {a.confidence != null && <span>Confidence: {a.confidence}</span>}
            </div>

            {a.problem_summary && <AnalysisSection title="Problem & Challenge" content={a.problem_summary} />}
            {a.method_summary && <AnalysisSection title="Method & Insight" content={a.method_summary} />}
            {a.core_intuition && <AnalysisSection title="Core Intuition" content={a.core_intuition} />}

            {a.changed_slots && a.changed_slots.length > 0 && (
              <div>
                <p className="text-sm font-medium text-gray-700 mb-1">Changed Slots:</p>
                <div className="flex gap-1">
                  {a.changed_slots.map((s) => <span key={s} className="px-2 py-0.5 bg-yellow-100 text-yellow-800 rounded text-xs">{s}</span>)}
                </div>
              </div>
            )}

            <div className="flex gap-4 text-xs text-gray-500">
              {a.is_plugin_patch != null && <span>Plugin patch: {a.is_plugin_patch ? "Yes" : "No"}</span>}
              {a.worth_deep_read != null && <span>Worth deep read: {a.worth_deep_read ? "Yes" : "No"}</span>}
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-400">No analysis yet. Click L3 Skim or L4 Deep to analyze.</p>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-1">{title}</h3>
      {children}
    </div>
  );
}

function ScoreBox({ label, value }: { label: string; value: number | string | null }) {
  const display = typeof value === "number" ? value.toFixed(2) : (value ?? "—");
  return (
    <div className="bg-white border rounded p-2 text-center">
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-lg font-semibold text-gray-800">{display}</p>
    </div>
  );
}

function AnalysisSection({ title, content }: { title: string; content: string }) {
  return (
    <div className="bg-gray-50 rounded p-3">
      <p className="text-sm font-medium text-gray-700 mb-1">{title}</p>
      <p className="text-sm text-gray-600 whitespace-pre-wrap">{content}</p>
    </div>
  );
}
