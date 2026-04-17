"use client";

import Link from "next/link";
import type { PaperBrief } from "@/lib/api";

const stateColors: Record<string, string> = {
  checked: "bg-green-100 text-green-800",
  l4_deep: "bg-green-100 text-green-800",
  l3_skimmed: "bg-blue-100 text-blue-800",
  l2_parsed: "bg-yellow-100 text-yellow-800",
  downloaded: "bg-gray-100 text-gray-800",
  wait: "bg-gray-100 text-gray-600",
  ephemeral_received: "bg-purple-100 text-purple-800",
  canonicalized: "bg-purple-100 text-purple-800",
};

const importanceColors: Record<string, string> = {
  S: "bg-red-500 text-white",
  A: "bg-orange-500 text-white",
  B: "bg-yellow-500 text-white",
  C: "bg-gray-400 text-white",
  D: "bg-gray-300 text-gray-600",
};

export default function PaperCard({ paper }: { paper: PaperBrief }) {
  return (
    <Link href={`/papers/${paper.id}`}>
      <div className="border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer bg-white">
        <div className="flex items-start justify-between gap-2 mb-2">
          <h3 className="font-medium text-sm leading-tight line-clamp-2">{paper.title}</h3>
          <div className="flex gap-1 shrink-0">
            {paper.importance && (
              <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${importanceColors[paper.importance] || ""}`}>
                {paper.importance}
              </span>
            )}
            <span className={`px-1.5 py-0.5 rounded text-xs ${stateColors[paper.state] || "bg-gray-100"}`}>
              {paper.state}
            </span>
          </div>
        </div>

        <div className="text-xs text-gray-500 mb-2">
          {paper.venue && <span>{paper.venue}</span>}
          {paper.year && <span> {paper.year}</span>}
          {" · "}
          <span>{paper.category}</span>
        </div>

        {paper.core_operator && (
          <p className="text-xs text-gray-600 line-clamp-2 mb-2">{paper.core_operator}</p>
        )}

        <div className="flex items-center gap-3 text-xs text-gray-400">
          {paper.keep_score != null && (
            <span>keep: {paper.keep_score.toFixed(2)}</span>
          )}
          {paper.structurality_score != null && (
            <span>struct: {paper.structurality_score.toFixed(2)}</span>
          )}
          {paper.is_ephemeral && (
            <span className="text-purple-500">ephemeral</span>
          )}
        </div>

        {paper.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {paper.tags.slice(0, 4).map((tag) => (
              <span key={tag} className="px-1.5 py-0.5 bg-gray-100 rounded text-xs text-gray-500">
                {tag}
              </span>
            ))}
            {paper.tags.length > 4 && (
              <span className="text-xs text-gray-400">+{paper.tags.length - 4}</span>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}
