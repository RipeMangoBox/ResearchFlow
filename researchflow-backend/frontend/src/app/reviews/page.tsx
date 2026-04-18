"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface ReviewTask {
  id: string;
  target_type: string;
  target_id: string;
  task_type: string;
  status: string;
  priority: number;
  assigned_to: string | null;
  notes: string | null;
  created_at: string | null;
}

interface Stats {
  by_status: Record<string, number>;
  by_target_type: Record<string, number>;
  total_pending: number;
}

export default function ReviewsPage() {
  const [tasks, setTasks] = useState<ReviewTask[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [filter, setFilter] = useState<string>("pending");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [filter]);

  async function loadData() {
    setLoading(true);
    try {
      const [queueData, statsData] = await Promise.all([
        api.getReviewQueue(filter),
        api.getReviewStats(),
      ]);
      setTasks(queueData.tasks || []);
      setStats(statsData);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }

  async function handleApprove(taskId: string) {
    try {
      await api.approveReview(taskId, "web_user");
      loadData();
    } catch (e) {
      alert(`Error: ${e}`);
    }
  }

  async function handleReject(taskId: string) {
    const reason = prompt("Rejection reason:");
    if (reason === null) return;
    try {
      await api.rejectReview(taskId, "web_user", reason);
      loadData();
    } catch (e) {
      alert(`Error: ${e}`);
    }
  }

  const priorityColor = (p: number) => {
    if (p <= 2) return "text-red-600 font-bold";
    if (p <= 3) return "text-yellow-600";
    return "text-gray-500";
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Review Queue</h1>

      {stats && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-yellow-50 border border-yellow-200 rounded p-3 text-center">
            <div className="text-2xl font-bold text-yellow-700">{stats.total_pending}</div>
            <div className="text-sm text-yellow-600">Pending</div>
          </div>
          {Object.entries(stats.by_target_type).map(([type, count]) => (
            <div key={type} className="bg-gray-50 border rounded p-3 text-center">
              <div className="text-xl font-semibold">{count}</div>
              <div className="text-sm text-gray-500">{type}</div>
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2 mb-4">
        {["pending", "in_progress", "approved", "rejected"].map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`px-3 py-1 rounded text-sm ${
              filter === s ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {s} {stats?.by_status[s] ? `(${stats.by_status[s]})` : ""}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="text-gray-400 py-8 text-center">Loading...</div>
      ) : tasks.length === 0 ? (
        <div className="text-gray-400 py-8 text-center">No tasks with status &quot;{filter}&quot;</div>
      ) : (
        <div className="space-y-3">
          {tasks.map((task) => (
            <div key={task.id} className="border rounded-lg p-4 hover:bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className={`text-xs ${priorityColor(task.priority)}`}>P{task.priority}</span>
                  <span className="bg-blue-100 text-blue-700 text-xs px-2 py-0.5 rounded">{task.target_type}</span>
                  <span className="text-xs text-gray-400">{task.task_type}</span>
                </div>
                {task.status === "pending" && (
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleApprove(task.id)}
                      className="text-xs bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700"
                    >
                      Approve
                    </button>
                    <button
                      onClick={() => handleReject(task.id)}
                      className="text-xs bg-red-600 text-white px-3 py-1 rounded hover:bg-red-700"
                    >
                      Reject
                    </button>
                  </div>
                )}
              </div>
              {task.notes && <p className="text-sm text-gray-600 mt-2">{task.notes}</p>}
              <div className="text-xs text-gray-400 mt-1">
                {task.created_at ? new Date(task.created_at).toLocaleString() : ""} &middot; {task.id.slice(0, 8)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
