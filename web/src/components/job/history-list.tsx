"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { api } from "@/lib/api/browser";

export function HistoryList() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["jobs"], queryFn: api.listJobs, staleTime: 0 });
  if (q.isError) return <p role="alert">Could not load your counts: {(q.error as Error).message}</p>;
  if (!q.data) return <p role="status">Loading…</p>;
  const jobs = [...q.data.jobs].sort((a, b) => b.created_at - a.created_at);
  if (jobs.length === 0) {
    return (
      <div data-testid="history-empty">
        <p>No counts yet.</p>
        <Button asChild className="mt-3">
          <Link href="/new">Count something</Link>
        </Button>
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-2" data-testid="history">
      {jobs.map((j) => (
        <li key={j.id} className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card p-3">
          <Link href={j.status === "draft" ? "/new" : `/jobs/${j.id}`} className="min-w-0 flex-1 underline-offset-4 hover:underline">
            <span className="font-medium">{j.prompt}</span>
            <span className="block text-sm text-muted-foreground">
              {new Date(j.created_at * 1000).toLocaleString()} · {j.status}
            </span>
          </Link>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            aria-label={`Delete ${j.prompt}`}
            disabled={j.status === "running"}
            onClick={async () => {
              if (!window.confirm("Delete this count and its photos?")) return;
              await api.deleteJob(j.id).catch(() => undefined);
              void qc.invalidateQueries({ queryKey: ["jobs"] });
            }}
          >
            Delete
          </Button>
        </li>
      ))}
    </ul>
  );
}
