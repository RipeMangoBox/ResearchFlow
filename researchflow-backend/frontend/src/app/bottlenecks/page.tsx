"use client";

import { useState } from "react";
import { api } from "@/lib/api";

export default function BottlenecksPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function handleSearch() {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const data = await api.intentQuery(query, "bottleneck");
      setResult(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Bottleneck Browser</h1>
      <p className="text-gray-500 text-sm mb-6">
        Search for research bottlenecks — what&apos;s blocking progress in a direction. Shows project-level focus, paper-level claims, and related delta cards.
      </p>

      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. reward shaping, temporal consistency, hallucination..."
          className="flex-1 border rounded px-3 py-2 text-sm"
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {result && (
        <div className="space-y-8">
          {result.project_focus?.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold mb-3 text-purple-700">
                Project Focus ({result.project_focus.length})
              </h2>
              <div className="space-y-2">
                {result.project_focus.map((f: any, i: number) => (
                  <div key={i} className="border border-purple-200 bg-purple-50 rounded-lg p-3">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{f.title}</span>
                      <span className="text-xs text-purple-600">P{f.priority}</span>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{f.description}</p>
                    {f.negative_constraints?.length > 0 && (
                      <div className="mt-1 flex gap-1">
                        {f.negative_constraints.map((c: string, j: number) => (
                          <span key={j} className="text-xs bg-red-100 text-red-600 px-1.5 rounded">{c}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {result.paper_claims?.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold mb-3 text-orange-700">
                Paper Claims ({result.paper_claims.length})
              </h2>
              <div className="space-y-2">
                {result.paper_claims.map((c: any, i: number) => (
                  <div key={i} className="border rounded-lg p-3 hover:bg-gray-50">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{c.paper_title}</span>
                      {c.is_fundamental && (
                        <span className="text-xs bg-red-200 text-red-700 px-1.5 rounded">fundamental</span>
                      )}
                      <span className="text-xs text-gray-400">conf={c.confidence?.toFixed(2)}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      <span className="text-orange-600">{c.bottleneck_title}</span> &mdash; {c.claim_text}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {result.delta_cards?.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold mb-3 text-blue-700">
                Related Delta Cards ({result.delta_cards.length})
              </h2>
              <div className="space-y-2">
                {result.delta_cards.map((d: any, i: number) => (
                  <div key={i} className="border rounded-lg p-3 hover:bg-gray-50">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{d.paper_title}</span>
                      {d.structurality_score != null && (
                        <span className="text-xs text-gray-400">struct={d.structurality_score.toFixed(2)}</span>
                      )}
                    </div>
                    <p className="text-xs text-gray-600 mt-1">{d.delta_statement}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {!result.project_focus?.length && !result.paper_claims?.length && !result.delta_cards?.length && (
            <div className="text-gray-400 text-center py-8">No results found for &quot;{query}&quot;</div>
          )}
        </div>
      )}
    </div>
  );
}
