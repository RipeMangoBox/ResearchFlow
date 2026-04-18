"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface GraphNode {
  id: string;
  label: string;
  title?: string;
  group: string;
  size: number;
  category?: string;
}

interface GraphEdge {
  from: string;
  to: string;
  label?: string;
  arrows?: string;
  color?: string;
  dashes?: boolean | number[];
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  groups: Record<string, { color: string; shape: string }>;
}

export default function GraphPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [limit, setLimit] = useState(200);
  const networkRef = useRef<any>(null);

  useEffect(() => {
    loadData();
  }, [limit]);

  async function loadData() {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/graph/vis-data?limit=${limit}`);
      const json = await res.json();
      setData(json);
    } catch (e) {
      console.error("Failed to load graph data:", e);
    }
    setLoading(false);
  }

  useEffect(() => {
    if (!data || !containerRef.current) return;

    // Dynamically load vis-network from CDN
    const script = document.createElement("script");
    script.src = "https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js";
    script.onload = () => {
      const vis = (window as any).vis;
      if (!vis) return;

      const groupColors: Record<string, any> = {};
      for (const [key, val] of Object.entries(data.groups)) {
        groupColors[key] = {
          color: { background: val.color, border: val.color, highlight: { background: val.color, border: "#333" } },
          shape: val.shape === "diamond" ? "diamond" : val.shape === "triangle" ? "triangle" : val.shape === "square" ? "square" : "dot",
          font: { size: 11, color: "#333" },
        };
      }

      const nodes = new vis.DataSet(
        data.nodes.map((n) => ({
          ...n,
          font: { size: Math.max(9, n.size * 0.8) },
          color: data.groups[n.group]?.color ? {
            background: data.groups[n.group].color,
            border: data.groups[n.group].color,
          } : undefined,
          shape: data.groups[n.group]?.shape === "diamond" ? "diamond"
            : data.groups[n.group]?.shape === "triangle" ? "triangle"
            : data.groups[n.group]?.shape === "square" ? "square"
            : "dot",
        }))
      );

      const edges = new vis.DataSet(
        data.edges.map((e, i) => ({
          ...e,
          id: `edge-${i}`,
          font: { size: 9, color: "#999", strokeWidth: 0 },
          smooth: { type: "continuous" },
        }))
      );

      const options = {
        physics: {
          solver: "forceAtlas2Based",
          forceAtlas2Based: { gravitationalConstant: -30, centralGravity: 0.005, springLength: 120 },
          stabilization: { iterations: 150 },
        },
        interaction: { hover: true, tooltipDelay: 200, multiselect: true },
        edges: { arrows: { to: { enabled: false, scaleFactor: 0.5 } }, width: 1 },
        groups: groupColors,
      };

      const network = new vis.Network(containerRef.current, { nodes, edges }, options);
      networkRef.current = network;

      network.on("click", (params: any) => {
        if (params.nodes.length > 0) {
          const nodeId = params.nodes[0];
          const node = data.nodes.find((n) => n.id === nodeId);
          setSelectedNode(node || null);
        } else {
          setSelectedNode(null);
        }
      });
    };
    document.head.appendChild(script);

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [data]);

  const groupCounts = data
    ? Object.entries(
        data.nodes.reduce((acc, n) => {
          acc[n.group] = (acc[n.group] || 0) + 1;
          return acc;
        }, {} as Record<string, number>)
      )
    : [];

  return (
    <div className="h-[calc(100vh-80px)] flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 bg-white border-b">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold">Knowledge Graph</h1>
          {groupCounts.map(([group, count]) => (
            <span
              key={group}
              className="text-xs px-2 py-0.5 rounded"
              style={{ backgroundColor: data?.groups[group]?.color + "22", color: data?.groups[group]?.color }}
            >
              {group}: {count}
            </span>
          ))}
          <span className="text-xs text-gray-400">{data?.edges.length || 0} edges</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500">Max nodes:</label>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="text-xs border rounded px-2 py-1"
          >
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
            <option value={500}>500</option>
          </select>
          <button onClick={loadData} className="text-xs bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700">
            Refresh
          </button>
        </div>
      </div>

      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
            <span className="text-gray-400">Loading graph...</span>
          </div>
        )}
        <div ref={containerRef} className="w-full h-full" />

        {selectedNode && (
          <div className="absolute top-4 right-4 bg-white border rounded-lg shadow-lg p-4 w-72 z-20">
            <div className="flex items-center justify-between mb-2">
              <span
                className="text-xs px-2 py-0.5 rounded text-white"
                style={{ backgroundColor: data?.groups[selectedNode.group]?.color }}
              >
                {selectedNode.group}
              </span>
              <button onClick={() => setSelectedNode(null)} className="text-gray-400 hover:text-gray-600 text-sm">
                &times;
              </button>
            </div>
            <h3 className="font-medium text-sm">{selectedNode.label}</h3>
            {selectedNode.title && (
              <p className="text-xs text-gray-500 mt-1 whitespace-pre-line">{selectedNode.title}</p>
            )}
            {selectedNode.category && (
              <p className="text-xs text-gray-400 mt-1">Category: {selectedNode.category}</p>
            )}
            {selectedNode.id.startsWith("paper:") && (
              <a
                href={`/papers/${selectedNode.id.replace("paper:", "")}`}
                className="text-xs text-blue-600 hover:underline mt-2 inline-block"
              >
                View paper detail &rarr;
              </a>
            )}
          </div>
        )}
      </div>

      <div className="px-4 py-2 bg-gray-50 border-t text-xs text-gray-400 flex gap-4">
        <span>Click node for details</span>
        <span>Scroll to zoom</span>
        <span>Drag to pan</span>
        <span>
          Legend:
          <span className="ml-1" style={{ color: "#3498db" }}>● Paper</span>
          <span className="ml-2" style={{ color: "#2ecc71" }}>◆ Mechanism</span>
          <span className="ml-2" style={{ color: "#9b59b6" }}>▲ Paradigm</span>
          <span className="ml-2" style={{ color: "#e74c3c" }}>■ Bottleneck</span>
        </span>
      </div>
    </div>
  );
}
