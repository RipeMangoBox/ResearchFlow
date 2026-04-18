"use client";

import { useState } from "react";
import { api } from "@/lib/api";

interface LineageNode {
  delta_card_id: string;
  paper_id: string;
  title: string;
  delta_statement: string;
  lineage_depth: number;
  downstream_count: number;
  is_established_baseline: boolean;
  depth_in_tree: number;
}

interface LineageResult {
  root: LineageNode;
  ancestors: LineageNode[];
  descendants: LineageNode[];
}

export default function LineagePage() {
  const [paperId, setPaperId] = useState("");
  const [result, setResult] = useState<LineageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch() {
    if (!paperId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getLineage(paperId.trim());
      if (data.error) {
        setError(data.error);
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (e: any) {
      setError(e.message);
      setResult(null);
    }
    setLoading(false);
  }

  function renderNode(node: LineageNode, role: string) {
    return (
      <div
        key={node.delta_card_id}
        className={`border rounded-lg p-3 ${
          node.is_established_baseline ? "border-green-400 bg-green-50" : "border-gray-200"
        }`}
        style={{ marginLeft: `${node.depth_in_tree * 24}px` }}
      >
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">{role}</span>
          {node.is_established_baseline && (
            <span className="text-xs bg-green-200 text-green-800 px-1.5 rounded">baseline</span>
          )}
          <span className="text-xs text-gray-400">depth={node.lineage_depth}</span>
          <span className="text-xs text-gray-400">downstream={node.downstream_count}</span>
        </div>
        <h3 className="font-medium text-sm mt-1">{node.title}</h3>
        <p className="text-xs text-gray-500 mt-1">{node.delta_statement.slice(0, 200)}</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Method Lineage Explorer</h1>
      <p className="text-gray-500 text-sm mb-6">
        Enter a paper ID to visualize its method evolution DAG — ancestors (what it builds on) and descendants (what builds on it).
      </p>

      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={paperId}
          onChange={(e) => setPaperId(e.target.value)}
          placeholder="Paper UUID"
          className="flex-1 border rounded px-3 py-2 text-sm"
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Loading..." : "Explore"}
        </button>
      </div>

      {error && <div className="text-red-600 text-sm mb-4">{error}</div>}

      {result && (
        <div className="space-y-6">
          {result.ancestors.length > 0 && (
            <div>
              <h2 className="text-lg font-semibold mb-2 text-gray-700">Ancestors (builds on)</h2>
              <div className="space-y-2">
                {result.ancestors.map((n) => renderNode(n, "ancestor"))}
              </div>
            </div>
          )}

          <div>
            <h2 className="text-lg font-semibold mb-2 text-blue-700">Current Paper</h2>
            {renderNode(result.root, "root")}
          </div>

          {result.descendants.length > 0 && (
            <div>
              <h2 className="text-lg font-semibold mb-2 text-gray-700">Descendants (built on this)</h2>
              <div className="space-y-2">
                {result.descendants.map((n) => renderNode(n, "descendant"))}
              </div>
            </div>
          )}

          {result.ancestors.length === 0 && result.descendants.length === 0 && (
            <div className="text-gray-400 text-center py-4">No lineage edges found for this paper.</div>
          )}
        </div>
      )}
    </div>
  );
}
