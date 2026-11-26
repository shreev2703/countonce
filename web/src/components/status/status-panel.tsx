"use client";

import { useQuery } from "@tanstack/react-query";

import { Card } from "@/components/ui/card";
import { api } from "@/lib/api/browser";

const seconds = (s: number | null) => (s === null ? "unknown" : s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`);
const OUTCOME: Record<string, string> = { done: "Finished", failed: "Failed", dead_letter: "Failed after retries" };

export function StatusPanel() {
  const q = useQuery({ queryKey: ["status"], queryFn: api.status, refetchInterval: 10_000, retry: false });
  if (q.isPending) return <p className="mt-6 text-muted-foreground">Checking…</p>;
  if (q.isError) {
    return (
      <Card role="alert" className="mt-6" data-testid="status-down">
        <p className="font-medium">The CountOnce API is not reachable.</p>
        <p className="text-sm text-muted-foreground">New counts cannot be started. The demo gallery still works.</p>
      </Card>
    );
  }
  const s = q.data;
  const stats = [
    ["Worker", s.workers_online > 0 ? "Online" : "Offline"],
    ["Jobs waiting", String(s.queue.queued)],
    ["Jobs running", String(s.queue.running)],
  ];
  return (
    <div data-testid="status-panel">
      <ul className="mt-6 grid gap-4 sm:grid-cols-3">
        {stats.map(([label, value]) => (
          <li key={label}>
            <Card>
              <p className="text-sm text-muted-foreground">{label}</p>
              <p className="tabular text-3xl font-semibold">{value}</p>
            </Card>
          </li>
        ))}
      </ul>
      {s.processing_paused && (
        <p role="status" className="mt-4 text-sm">
          Processing is paused. Submitted jobs stay queued and start when the worker comes back.
        </p>
      )}
      <h2 className="mt-8 text-xl font-semibold">Recent jobs</h2>
      {s.recent.length === 0 ? (
        <p className="mt-2 text-muted-foreground">No finished jobs yet.</p>
      ) : (
        <table className="tabular mt-2 w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left">
              <th scope="col" className="py-2">Finished</th>
              <th scope="col" className="py-2">Outcome</th>
              <th scope="col" className="py-2">Waited</th>
              <th scope="col" className="py-2">Processed in</th>
            </tr>
          </thead>
          <tbody>
            {s.recent.map((j) => (
              <tr key={j.finished_at} className="border-b border-border">
                <td className="py-2">{new Date(j.finished_at * 1000).toLocaleString()}</td>
                <td className="py-2">{OUTCOME[j.status] ?? j.status}</td>
                <td className="py-2">{seconds(j.wait_s)}</td>
                <td className="py-2">{seconds(j.run_s)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
