"use client";

import { useEffect, useState } from "react";
import { api, type DigestInfo } from "@/lib/api";

export default function DigestsPage() {
  const [digests, setDigests] = useState<Record<string, DigestInfo | null>>({});
  const [activeTab, setActiveTab] = useState("day");
  const [generating, setGenerating] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.latestDigests().then(setDigests).catch(console.error).finally(() => setLoading(false));
  }, []);

  async function generate(type: string) {
    setGenerating(type);
    try {
      const res = await api.generateDigest(type);
      setDigests((prev) => ({
        ...prev,
        [type]: { id: "", period_start: res.period_start, period_end: res.period_end, content: res.content },
      }));
      setActiveTab(type);
    } catch (e) { console.error(e); }
    setGenerating("");
  }

  if (loading) return <div className="text-center py-20 text-gray-400">Loading...</div>;

  const tabs = [
    { key: "day", label: "Daily", desc: "What's new today" },
    { key: "week", label: "Weekly", desc: "Direction trends" },
    { key: "month", label: "Monthly", desc: "Strategy review" },
  ];

  const current = digests[activeTab];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Research Digests</h1>

      <div className="flex gap-3">
        {tabs.map((tab) => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 rounded-lg text-sm transition-colors ${activeTab === tab.key ? "bg-blue-600 text-white" : "bg-white border text-gray-600 hover:bg-gray-50"}`}>
            <span className="font-medium">{tab.label}</span>
            <span className="block text-xs opacity-70">{tab.desc}</span>
          </button>
        ))}
      </div>

      <div className="bg-white border rounded-lg">
        <div className="flex items-center justify-between p-4 border-b">
          <div>
            <h2 className="font-semibold text-gray-800">
              {tabs.find((t) => t.key === activeTab)?.label} Digest
            </h2>
            {current && (
              <p className="text-xs text-gray-400">{current.period_start} to {current.period_end}</p>
            )}
          </div>
          <button onClick={() => generate(activeTab)} disabled={!!generating}
            className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50">
            {generating === activeTab ? "Generating..." : "Generate"}
          </button>
        </div>

        <div className="p-4">
          {current ? (
            <div className="prose prose-sm max-w-none whitespace-pre-wrap text-sm text-gray-700">
              {current.content}
            </div>
          ) : (
            <p className="text-sm text-gray-400 text-center py-8">
              No {activeTab} digest yet. Click Generate to create one.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
