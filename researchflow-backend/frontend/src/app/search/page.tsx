"use client";

import { useState } from "react";
import Link from "next/link";
import { api, type SearchResult } from "@/lib/api";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await api.hybridSearch(query, { category: category || undefined, limit: 20 });
      setResults(res.results);
      setSearched(true);
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Search Knowledge Base</h1>

      <form onSubmit={handleSearch} className="flex gap-3">
        <input
          type="text"
          className="flex-1 border rounded px-4 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
          placeholder="Search papers... (e.g., 'diffusion motion generation physics')"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select className="border rounded px-2 py-2 text-sm" value={category} onChange={(e) => setCategory(e.target.value)}>
          <option value="">All categories</option>
          <option value="Motion_Generation_Text_Speech_Music_Driven">Motion Generation</option>
          <option value="Human_Object_Interaction">HOI</option>
          <option value="Human_Human_Interaction">HHI</option>
          <option value="Human_Scene_Interaction">HSI</option>
        </select>
        <button type="submit" disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm">
          {loading ? "..." : "Search"}
        </button>
      </form>

      {searched && (
        <p className="text-sm text-gray-400">{results.length} results for &quot;{query}&quot;</p>
      )}

      <div className="space-y-3">
        {results.map((r) => (
          <Link key={r.paper_id} href={`/papers/${r.paper_id}`}>
            <div className="bg-white border rounded-lg p-4 hover:shadow-sm transition-shadow cursor-pointer">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <h3 className="font-medium text-sm text-gray-900">{r.title}</h3>
                  <div className="text-xs text-gray-500 mt-1">
                    {r.venue} {r.year} · {r.category.replace(/_/g, " ")}
                  </div>
                  {r.core_operator && (
                    <p className="text-xs text-gray-500 mt-1 line-clamp-1">{r.core_operator}</p>
                  )}
                  <div className="flex gap-1 mt-2">
                    {r.tags.slice(0, 3).map((t) => (
                      <span key={t} className="px-1.5 py-0.5 bg-gray-100 rounded text-xs text-gray-500">{t}</span>
                    ))}
                  </div>
                </div>
                <div className="text-right shrink-0 text-xs text-gray-400 space-y-1">
                  <div>score: <span className="font-mono font-medium text-gray-600">{r.combined_score.toFixed(3)}</span></div>
                  <div>text: {r.text_score.toFixed(3)}</div>
                  <div>vec: {r.vector_score.toFixed(3)}</div>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
