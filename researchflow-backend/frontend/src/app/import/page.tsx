"use client";

import { useState } from "react";
import { api } from "@/lib/api";

export default function ImportPage() {
  const [urls, setUrls] = useState("");
  const [category, setCategory] = useState("Uncategorized");
  const [ephemeral, setEphemeral] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function handleImport() {
    const lines = urls.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) return;
    setLoading(true);
    try {
      const items = lines.map((url) => ({ url }));
      const res = await api.importLinks(items, category);
      setResult(res);
    } catch (e: any) {
      setResult({ error: e.message });
    }
    setLoading(false);
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Import Papers</h1>

      <div className="bg-white border rounded-lg p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Paper URLs (one per line)</label>
          <textarea
            className="w-full border rounded p-3 text-sm font-mono h-40 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            placeholder={"https://arxiv.org/abs/2410.05260\nhttps://arxiv.org/abs/2312.00063\nhttps://arxiv.org/abs/2503.19901"}
            value={urls}
            onChange={(e) => setUrls(e.target.value)}
          />
        </div>

        <div className="flex gap-4 items-end">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
            <select className="border rounded px-3 py-2 text-sm" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option>Uncategorized</option>
              <option>Motion_Generation_Text_Speech_Music_Driven</option>
              <option>Human_Object_Interaction</option>
              <option>Human_Human_Interaction</option>
              <option>Human_Scene_Interaction</option>
            </select>
          </div>
          <label className="flex items-center gap-2 text-sm text-gray-600">
            <input type="checkbox" checked={ephemeral} onChange={(e) => setEphemeral(e.target.checked)} />
            Temporary (30-day expiry)
          </label>
        </div>

        <button
          onClick={handleImport}
          disabled={loading || !urls.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
        >
          {loading ? "Importing..." : "Import"}
        </button>
      </div>

      {result && !result.error && (
        <div className="bg-white border rounded-lg p-6">
          <h2 className="font-semibold text-gray-800 mb-3">
            Results: {result.created} created, {result.duplicates} duplicates, {result.errors} errors
          </h2>
          <div className="space-y-2">
            {result.items?.map((item: any, i: number) => (
              <div key={i} className={`text-sm p-2 rounded ${item.status === "created" ? "bg-green-50" : item.status === "duplicate" ? "bg-yellow-50" : "bg-red-50"}`}>
                <span className="font-medium">[{item.status}]</span> {item.title}
                {item.message && <span className="text-gray-400 ml-2">— {item.message}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
      {result?.error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">{result.error}</div>
      )}
    </div>
  );
}
