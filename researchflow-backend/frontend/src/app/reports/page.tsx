"use client";

import { useEffect, useState } from "react";
import { api, type PaperBrief } from "@/lib/api";

export default function ReportsPage() {
  const [papers, setPapers] = useState<PaperBrief[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [reportType, setReportType] = useState("briefing");
  const [topic, setTopic] = useState("");
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.listPapers({ size: "30", state: "checked", sort_by: "keep_score", sort_order: "desc" })
      .then((res) => setPapers(res.items))
      .catch(console.error);
  }, []);

  function toggle(id: string) {
    const next = new Set(selected);
    next.has(id) ? next.delete(id) : next.add(id);
    setSelected(next);
  }

  async function generate() {
    if (selected.size === 0) return;
    setLoading(true);
    setReport(null);
    try {
      const res = await api.generateReport([...selected], reportType, topic || undefined);
      setReport(res.content);
    } catch (e: any) {
      setReport(`Error: ${e.message}`);
    }
    setLoading(false);
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Generate Report</h1>

      <div className="flex gap-4 items-end">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Report Type</label>
          <select className="border rounded px-3 py-2 text-sm" value={reportType} onChange={(e) => setReportType(e.target.value)}>
            <option value="quick">Quick (30s)</option>
            <option value="briefing">Briefing (5min)</option>
            <option value="deep_compare">Deep Compare</option>
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-1">Topic (optional)</label>
          <input className="w-full border rounded px-3 py-2 text-sm" placeholder="e.g., diffusion-based motion generation" value={topic} onChange={(e) => setTopic(e.target.value)} />
        </div>
        <button onClick={generate} disabled={loading || selected.size === 0}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm">
          {loading ? "Generating..." : `Generate (${selected.size} papers)`}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Paper selection */}
        <div>
          <h2 className="font-semibold text-gray-800 mb-3">Select Papers ({selected.size} selected)</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {papers.map((p) => (
              <label key={p.id} className={`flex items-start gap-2 p-2 rounded cursor-pointer text-sm ${selected.has(p.id) ? "bg-blue-50 border-blue-200 border" : "bg-white border hover:bg-gray-50"}`}>
                <input type="checkbox" checked={selected.has(p.id)} onChange={() => toggle(p.id)} className="mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-800 line-clamp-1">{p.title}</p>
                  <p className="text-xs text-gray-400">{p.venue} {p.year} · keep={p.keep_score?.toFixed(2)}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Report output */}
        <div>
          <h2 className="font-semibold text-gray-800 mb-3">Report</h2>
          {report ? (
            <div className="bg-white border rounded-lg p-4 max-h-[600px] overflow-y-auto">
              <div className="prose prose-sm max-w-none whitespace-pre-wrap text-sm text-gray-700">
                {report}
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 border border-dashed rounded-lg p-8 text-center text-sm text-gray-400">
              Select papers and click Generate
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
