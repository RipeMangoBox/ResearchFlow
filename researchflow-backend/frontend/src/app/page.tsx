"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type PaperBrief, type DigestInfo } from "@/lib/api";
import PaperCard from "@/components/PaperCard";

export default function Dashboard() {
  const [stats, setStats] = useState<{ total: number; papers: PaperBrief[] } | null>(null);
  const [digests, setDigests] = useState<Record<string, DigestInfo | null>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [papersRes, digestRes] = await Promise.all([
          api.listPapers({ size: "6", sort_by: "updated_at", sort_order: "desc" }),
          api.latestDigests(),
        ]);
        setStats({ total: papersRes.total, papers: papersRes.items });
        setDigests(digestRes);
      } catch (e) {
        console.error("Dashboard load error:", e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div className="text-center py-20 text-gray-400">Loading...</div>;

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-1">Dashboard</h1>
        <p className="text-gray-500 text-sm">Research operating system overview</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Papers" value={stats?.total ?? 0} />
        <StatCard label="Daily Digest" value={digests.day ? "Ready" : "—"} sub={digests.day?.period_start} />
        <StatCard label="Weekly Digest" value={digests.week ? "Ready" : "—"} sub={digests.week?.period_start} />
        <StatCard label="Monthly Digest" value={digests.month ? "Ready" : "—"} sub={digests.month?.period_start} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <ActionBtn href="/import" label="Import Papers" desc="Links or PDFs" />
        <ActionBtn href="/search" label="Search KB" desc="Hybrid search" />
        <ActionBtn href="/reports" label="Reports" desc="30s / 5min / deep" />
        <ActionBtn href="/digests" label="Digests" desc="Day / week / month" />
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-gray-800">Recent Papers</h2>
          <Link href="/papers" className="text-sm text-blue-600 hover:underline">View all</Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {stats?.papers.map((p) => <PaperCard key={p.id} paper={p} />)}
        </div>
      </div>

      {digests.day && (
        <div>
          <h2 className="text-lg font-semibold text-gray-800 mb-3">Latest Daily Digest</h2>
          <div className="bg-white border rounded-lg p-4">
            <p className="text-xs text-gray-400 mb-2">{digests.day.period_start}</p>
            <div className="text-sm text-gray-700 whitespace-pre-wrap line-clamp-6">{digests.day.content}</div>
            <Link href="/digests" className="text-sm text-blue-600 hover:underline mt-2 inline-block">Read full</Link>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white border rounded-lg p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}

function ActionBtn({ href, label, desc }: { href: string; label: string; desc: string }) {
  return (
    <Link href={href}>
      <div className="bg-white border rounded-lg p-3 hover:shadow-sm transition-shadow cursor-pointer text-center">
        <p className="font-medium text-sm text-gray-800">{label}</p>
        <p className="text-xs text-gray-400">{desc}</p>
      </div>
    </Link>
  );
}
