"use client";

import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api/browser";

/** Shown while a job waits and no worker has sent a heartbeat (the owner's laptop is off). */
export function WorkerBanner({ waiting }: { waiting: boolean }) {
  const q = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 10_000, enabled: waiting });
  if (!waiting || !q.data?.processing_paused) return null;
  return (
    <div role="status" data-testid="paused-banner" className="rounded-md border border-status-duplicate bg-card p-3 text-sm">
      <p className="font-medium">Processing is paused right now.</p>
      <p className="text-muted-foreground">
        Your photos are saved and queued. Counting starts by itself when the worker comes back online, and this page
        updates. You can close it and return from History.
      </p>
    </div>
  );
}
