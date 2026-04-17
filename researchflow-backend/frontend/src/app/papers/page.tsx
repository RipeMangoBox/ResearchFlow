"use client";

import { useEffect, useState } from "react";
import { api, type PaperBrief, type PaperListResponse } from "@/lib/api";
import PaperCard from "@/components/PaperCard";

export default function PapersPage() {
  const [data, setData] = useState<PaperListResponse | null>(null);
  const [filters, setFilters] = useState({ category: "", state: "", sort_by: "updated_at", page: "1", size: "12" });
  const [loading, setLoading] = useState(true);

  async function load(f = filters) {
    setLoading(true);
    try {
      const params: Record<string, string> = { ...f };
      Object.keys(params).forEach((k) => { if (!params[k]) delete params[k]; });
      const res = await api.listPapers(params);
      setData(res);
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  function setFilter(key: string, val: string) {
    const next = { ...filters, [key]: val, page: "1" };
    setFilters(next);
    load(next);
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Papers</h1>

      <div className="flex flex-wrap gap-3 items-center">
        <select className="border rounded px-2 py-1 text-sm" value={filters.category} onChange={(e) => setFilter("category", e.target.value)}>
          <option value="">All categories</option>
          {["Motion_Generation_Text_Speech_Music_Driven","Human_Object_Interaction","Human_Human_Interaction","Human_Scene_Interaction","Multimodal_Interleaving_Reasoning","Motion_Controlled_ImageVideo_Generation"].map((c) => (
            <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
          ))}
        </select>
        <select className="border rounded px-2 py-1 text-sm" value={filters.state} onChange={(e) => setFilter("state", e.target.value)}>
          <option value="">All states</option>
          {["checked","l4_deep","l3_skimmed","l2_parsed","downloaded","wait"].map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select className="border rounded px-2 py-1 text-sm" value={filters.sort_by} onChange={(e) => setFilter("sort_by", e.target.value)}>
          <option value="updated_at">Recent</option>
          <option value="keep_score">Keep score</option>
          <option value="structurality_score">Structurality</option>
          <option value="year">Year</option>
        </select>
        {data && <span className="text-sm text-gray-400">{data.total} papers</span>}
      </div>

      {loading ? (
        <div className="text-center py-10 text-gray-400">Loading...</div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data?.items.map((p) => <PaperCard key={p.id} paper={p} />)}
          </div>
          {data && data.pages > 1 && (
            <div className="flex justify-center gap-2 pt-4">
              {Array.from({ length: Math.min(data.pages, 10) }, (_, i) => (
                <button key={i} onClick={() => setFilter("page", String(i + 1))}
                  className={`px-3 py-1 rounded text-sm ${data.page === i + 1 ? "bg-blue-600 text-white" : "bg-white border text-gray-600 hover:bg-gray-50"}`}>
                  {i + 1}
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
