"use client";

import { useState } from "react";
import Link from "next/link";
import { api, type SearchResult } from "@/lib/api";

type SearchMode = "hybrid" | "intent";

const INTENT_OPTIONS = [
  { value: "", label: "Auto-detect" },
  { value: "bottleneck", label: "Bottleneck (what's blocking?)" },
  { value: "mechanism", label: "Mechanism (what approaches?)" },
  { value: "lineage", label: "Lineage (how did it evolve?)" },
  { value: "evidence", label: "Evidence (where's the proof?)" },
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [mode, setMode] = useState<SearchMode>("intent");
  const [intent, setIntent] = useState("");
  const [hybridResults, setHybridResults] = useState<SearchResult[]>([]);
  const [intentResult, setIntentResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setSearched(true);
    try {
      if (mode === "hybrid") {
        const res = await api.hybridSearch(query, { category: category || undefined, limit: 20 });
        setHybridResults(res.results);
        setIntentResult(null);
      } else {
        const res = await api.intentQuery(query, intent || undefined);
        setIntentResult(res);
        setHybridResults([]);
      }
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Search Knowledge Base</h1>

      <form onSubmit={handleSearch} className="space-y-3">
        <div className="flex gap-3">
          <input
            type="text"
            className="flex-1 border rounded px-4 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
            placeholder={mode === "intent"
              ? "Ask a question... (e.g., 'reward shaping bottleneck', 'diffusion method evolution')"
              : "Search papers... (keyword + semantic)"}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm">
            {loading ? "..." : "Search"}
          </button>
        </div>

        <div className="flex items-center gap-4 text-sm">
          <div className="flex bg-gray-100 rounded p-0.5">
            <button
              type="button"
              onClick={() => setMode("intent")}
              className={`px-3 py-1 rounded text-xs ${mode === "intent" ? "bg-white shadow text-blue-600" : "text-gray-500"}`}
            >
              Intent Query
            </button>
            <button
              type="button"
              onClick={() => setMode("hybrid")}
              className={`px-3 py-1 rounded text-xs ${mode === "hybrid" ? "bg-white shadow text-blue-600" : "text-gray-500"}`}
            >
              Hybrid Search
            </button>
          </div>

          {mode === "intent" && (
            <select className="border rounded px-2 py-1 text-xs" value={intent} onChange={(e) => setIntent(e.target.value)}>
              {INTENT_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          )}

          <select className="border rounded px-2 py-1 text-xs" value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">All categories</option>
            <option value="Motion_Generation_Text_Speech_Music_Driven">Motion Generation</option>
            <option value="Human_Object_Interaction">HOI</option>
            <option value="Human_Human_Interaction">HHI</option>
            <option value="Human_Scene_Interaction">HSI</option>
          </select>
        </div>
      </form>

      {/* Intent query results */}
      {intentResult && (
        <div className="space-y-6">
          <div className="flex items-center gap-2 text-sm">
            <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded text-xs">Intent: {intentResult.intent}</span>
            <span className="text-gray-400">for &quot;{intentResult.query}&quot;</span>
          </div>

          {/* Bottleneck results */}
          {intentResult.project_focus?.length > 0 && (
            <Section title="Project Focus" color="purple" count={intentResult.project_focus.length}>
              {intentResult.project_focus.map((f: any, i: number) => (
                <ResultCard key={i} title={f.title} subtitle={f.description} badge={`P${f.priority}`} badgeColor="purple" />
              ))}
            </Section>
          )}
          {intentResult.paper_claims?.length > 0 && (
            <Section title="Paper Claims" color="orange" count={intentResult.paper_claims.length}>
              {intentResult.paper_claims.map((c: any, i: number) => (
                <ResultCard key={i} title={c.paper_title} subtitle={`${c.bottleneck_title} — ${c.claim_text}`}
                  badge={c.is_fundamental ? "fundamental" : undefined} badgeColor="red"
                  meta={`conf: ${c.confidence?.toFixed(2) || "?"}`} />
              ))}
            </Section>
          )}

          {/* Mechanism results */}
          {intentResult.canonical_ideas?.length > 0 && (
            <Section title="Canonical Ideas" color="green" count={intentResult.canonical_ideas.length}>
              {intentResult.canonical_ideas.map((ci: any, i: number) => (
                <ResultCard key={i} title={ci.title} subtitle={ci.description}
                  meta={`${ci.contribution_count} contributions · ${ci.domain || ""}`} />
              ))}
            </Section>
          )}
          {intentResult.mechanism_families?.length > 0 && (
            <Section title="Mechanism Families" color="green" count={intentResult.mechanism_families.length}>
              {intentResult.mechanism_families.map((mf: any, i: number) => (
                <ResultCard key={i} title={mf.name} subtitle={mf.description} meta={mf.domain} />
              ))}
            </Section>
          )}
          {intentResult.paper_contributions?.length > 0 && (
            <Section title="Paper Contributions" color="blue" count={intentResult.paper_contributions.length}>
              {intentResult.paper_contributions.map((pc: any, i: number) => (
                <ResultCard key={i} title={pc.paper_title} subtitle={pc.delta_statement}
                  meta={`struct: ${pc.structurality_score?.toFixed(2) || "?"} · evidence: ${pc.evidence_count}`}
                  linkTo={`/papers/${pc.paper_id}`} />
              ))}
            </Section>
          )}

          {/* Lineage results */}
          {intentResult.lineage_trees?.length > 0 && (
            <Section title="Method Lineage" color="indigo" count={intentResult.lineage_trees.length}>
              {intentResult.lineage_trees.map((lt: any, i: number) => (
                <div key={i} className="border rounded-lg p-3 space-y-2">
                  <div className="font-medium text-sm">{lt.paper_title}</div>
                  <p className="text-xs text-gray-500">{lt.delta_statement}</p>
                  <div className="text-xs text-gray-400">
                    depth={lt.lineage_depth} · downstream={lt.downstream_count}
                    {lt.is_established_baseline && <span className="ml-1 text-green-600">baseline</span>}
                  </div>
                  {lt.ancestors?.length > 0 && (
                    <div className="pl-3 border-l-2 border-gray-200 space-y-1">
                      {lt.ancestors.map((a: any, j: number) => (
                        <div key={j} className="text-xs text-gray-500">
                          <span className="text-indigo-500">{a.relation_type}</span> {a.parent_paper_title}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </Section>
          )}

          {/* Evidence results */}
          {intentResult.evidence_units?.length > 0 && (
            <Section title="Evidence" color="amber" count={intentResult.evidence_units.length}>
              {intentResult.evidence_units.map((eu: any, i: number) => (
                <ResultCard key={i} title={eu.paper_title} subtitle={eu.claim}
                  badge={eu.basis} badgeColor={eu.basis === "experiment_backed" ? "green" : "gray"}
                  meta={`${eu.atom_type} · conf: ${eu.confidence?.toFixed(2) || "?"} · ${eu.source_section || ""}`} />
              ))}
            </Section>
          )}
          {intentResult.implementation_units?.length > 0 && (
            <Section title="Implementations" color="teal" count={intentResult.implementation_units.length}>
              {intentResult.implementation_units.map((iu: any, i: number) => (
                <ResultCard key={i} title={iu.paper_title} subtitle={iu.description || iu.class_or_function}
                  meta={iu.repo_url} />
              ))}
            </Section>
          )}

          {/* Delta cards (shared across intents) */}
          {intentResult.delta_cards?.length > 0 && (
            <Section title="Delta Cards" color="blue" count={intentResult.delta_cards.length}>
              {intentResult.delta_cards.map((dc: any, i: number) => (
                <ResultCard key={i} title={dc.paper_title} subtitle={dc.delta_statement}
                  meta={dc.structurality_score != null ? `struct: ${dc.structurality_score.toFixed(2)}` : undefined}
                  linkTo={dc.paper_id ? `/papers/${dc.paper_id}` : undefined} />
              ))}
            </Section>
          )}
        </div>
      )}

      {/* Hybrid search results */}
      {hybridResults.length > 0 && (
        <>
          <p className="text-sm text-gray-400">{hybridResults.length} results</p>
          <div className="space-y-3">
            {hybridResults.map((r) => (
              <Link key={r.paper_id} href={`/papers/${r.paper_id}`}>
                <div className="bg-white border rounded-lg p-4 hover:shadow-sm transition-shadow cursor-pointer">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <h3 className="font-medium text-sm text-gray-900">{r.title}</h3>
                      <div className="text-xs text-gray-500 mt-1">{r.venue} {r.year} · {r.category?.replace(/_/g, " ")}</div>
                      {r.core_operator && <p className="text-xs text-gray-500 mt-1 line-clamp-1">{r.core_operator}</p>}
                      <div className="flex gap-1 mt-2">
                        {r.tags?.slice(0, 3).map((t) => (
                          <span key={t} className="px-1.5 py-0.5 bg-gray-100 rounded text-xs text-gray-500">{t}</span>
                        ))}
                      </div>
                    </div>
                    <div className="text-right shrink-0 text-xs text-gray-400 space-y-1">
                      <div>score: <span className="font-mono font-medium text-gray-600">{r.combined_score.toFixed(3)}</span></div>
                      <div>text: {r.text_score.toFixed(3)} · vec: {r.vector_score.toFixed(3)}</div>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </>
      )}

      {searched && !loading && hybridResults.length === 0 && !intentResult && (
        <p className="text-gray-400 text-center py-8">No results found</p>
      )}
    </div>
  );
}

// ── Reusable components ──────────────────────────────────────────

function Section({ title, color, count, children }: { title: string; color: string; count: number; children: React.ReactNode }) {
  return (
    <section>
      <h2 className={`text-sm font-semibold mb-2 text-${color}-700`}>{title} ({count})</h2>
      <div className="space-y-2">{children}</div>
    </section>
  );
}

function ResultCard({ title, subtitle, badge, badgeColor, meta, linkTo }: {
  title: string; subtitle?: string; badge?: string; badgeColor?: string; meta?: string; linkTo?: string;
}) {
  const inner = (
    <div className="border rounded-lg p-3 hover:bg-gray-50 transition-colors">
      <div className="flex items-center gap-2">
        <span className="font-medium text-sm">{title}</span>
        {badge && (
          <span className={`text-xs px-1.5 py-0.5 rounded bg-${badgeColor || "gray"}-100 text-${badgeColor || "gray"}-700`}>{badge}</span>
        )}
      </div>
      {subtitle && <p className="text-xs text-gray-600 mt-1 line-clamp-2">{subtitle}</p>}
      {meta && <p className="text-xs text-gray-400 mt-1">{meta}</p>}
    </div>
  );
  if (linkTo) return <Link href={linkTo}>{inner}</Link>;
  return inner;
}
