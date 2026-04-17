"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface Direction {
  id: string;
  title: string;
  rationale: string | null;
  is_structural: boolean | null;
  estimated_cost: string | null;
  max_risk: string | null;
  confidence: number | null;
  has_feasibility_plan: boolean;
  source_topic?: string;
}

export default function DirectionsPage() {
  const [topic, setTopic] = useState("");
  const [category, setCategory] = useState("");
  const [directions, setDirections] = useState<Direction[]>([]);
  const [expandedPlan, setExpandedPlan] = useState<string | null>(null);
  const [expandingId, setExpandingId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<Direction[]>([]);

  useEffect(() => {
    fetch(`${API}/directions`).then(r => r.json()).then(setHistory).catch(console.error);
  }, []);

  async function propose() {
    if (!topic.trim()) return;
    setLoading(true);
    setDirections([]);
    setExpandedPlan(null);
    try {
      const res = await fetch(`${API}/directions/propose`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, category: category || undefined }),
      });
      const data = await res.json();
      setDirections(data.directions || []);
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  async function expand(id: string) {
    setExpandingId(id);
    setExpandedPlan(null);
    try {
      const res = await fetch(`${API}/directions/${id}/expand`, { method: "POST" });
      const data = await res.json();
      setExpandedPlan(data.feasibility_plan);
    } catch (e) { console.error(e); }
    setExpandingId(null);
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Research Directions</h1>

      <div className="bg-white border rounded-lg p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Describe your research interest</label>
          <textarea
            className="w-full border rounded p-3 text-sm h-24 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            placeholder="e.g., I want to explore physics-based human-object interaction using diffusion models, with a focus on real-time control..."
            value={topic} onChange={(e) => setTopic(e.target.value)}
          />
        </div>
        <div className="flex gap-4 items-end">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Category (optional)</label>
            <select className="border rounded px-3 py-2 text-sm" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="">Any</option>
              <option value="Motion_Generation_Text_Speech_Music_Driven">Motion Generation</option>
              <option value="Human_Object_Interaction">HOI</option>
              <option value="Human_Human_Interaction">HHI</option>
            </select>
          </div>
          <button onClick={propose} disabled={loading || !topic.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm">
            {loading ? "Proposing..." : "Propose Directions"}
          </button>
        </div>
      </div>

      {/* Direction cards */}
      {directions.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-800">Proposed Directions</h2>
          {directions.map((d) => (
            <div key={d.id} className="bg-white border rounded-lg p-5">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900">{d.title}</h3>
                  {d.rationale && <p className="text-sm text-gray-600 mt-1">{d.rationale}</p>}
                  <div className="flex gap-4 mt-3 text-xs text-gray-500">
                    {d.is_structural != null && (
                      <span className={`px-2 py-0.5 rounded ${d.is_structural ? "bg-green-100 text-green-700" : "bg-yellow-100 text-yellow-700"}`}>
                        {d.is_structural ? "Structural" : "Incremental"}
                      </span>
                    )}
                    {d.confidence != null && <span>Confidence: {d.confidence}</span>}
                    {d.estimated_cost && <span>Cost: {d.estimated_cost}</span>}
                    {d.max_risk && <span>Risk: {d.max_risk}</span>}
                  </div>
                </div>
                <button onClick={() => expand(d.id)} disabled={expandingId === d.id}
                  className="px-3 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-700 disabled:opacity-50 shrink-0">
                  {expandingId === d.id ? "..." : "Expand Plan"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Expanded feasibility plan */}
      {expandedPlan && (
        <div className="bg-white border rounded-lg p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-3">Feasibility Plan</h2>
          <div className="prose prose-sm max-w-none whitespace-pre-wrap text-sm text-gray-700">
            {expandedPlan}
          </div>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-gray-800 mb-3">Previous Directions</h2>
          <div className="space-y-2">
            {history.map((d: any) => (
              <div key={d.id} className="bg-white border rounded p-3 text-sm">
                <span className="font-medium">{d.title}</span>
                <span className="text-gray-400 ml-2">— {d.source_topic}</span>
                {d.has_feasibility_plan && <span className="ml-2 text-green-600 text-xs">has plan</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
