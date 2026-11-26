"use client";

import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { JobView as JobT } from "@/lib/api/browser";
import type { JobBundle } from "@/lib/server/job";
import { useJob } from "@/lib/job-events";

import { ResultPanel } from "./result-panel";
import { WorkerBanner } from "./worker-banner";

const STAGES: Record<string, string> = {
  queued: "Waiting for a worker",
  ingest: "Reading photos",
  detecting: "Finding objects in each photo",
  recon: "Working out where the photos were taken",
  reconstructing: "Working out where the photos were taken",
  matching: "Matching objects across photos",
  lifting: "Placing objects in 3D",
  counting: "Counting each object once",
  done: "Done",
};

export function JobView({ id, initial, initialResult, initialParsed, initialCorrections }: { id: string; initial?: JobT; initialResult?: JobBundle["result"]; initialParsed?: JobBundle["parsed"]; initialCorrections?: JobBundle["corrections"] }) {
  const { job, transport, error } = useJob(id, initial);
  if (!job) {
    return <p role="status">{error ?? "Loading…"}</p>;
  }
  const failed = job.status === "failed" || job.status === "dead_letter";
  const pct = Math.round(job.progress * 100);
  const stage = STAGES[job.stage ?? job.status] ?? job.stage ?? job.status;

  return (
    <div className="flex flex-col gap-4">
      <header>
        <h1 className="text-2xl font-semibold">Counting: {job.prompt}</h1>
        <p className="text-sm text-muted-foreground" data-testid="job-status" data-status={job.status}>
          Status: {job.status}
        </p>
      </header>
      <WorkerBanner waiting={job.status === "queued"} />

      {job.status !== "done" && !failed && (
        <Card className="flex flex-col gap-2">
          <div role="progressbar" aria-label="Progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div className="h-full bg-accent transition-[width]" style={{ width: `${pct}%` }} />
          </div>
          <p aria-live="polite" data-testid="stage" className="text-sm">
            {stage} ({pct}%)
          </p>
          <p className="text-xs text-muted-foreground" data-testid="transport" data-transport={transport}>
            {transport === "polling" ? "Live updates paused; checking every few seconds." : "Live updates on."}
          </p>
        </Card>
      )}

      {failed && (
        <div role="alert" className="rounded-md border border-status-rejected p-4" data-testid="job-failed">
          <p className="font-medium">This count did not finish.</p>
          <p className="text-sm">{job.error_message}</p>
          {job.tip && <p className="text-sm text-muted-foreground">{job.tip}</p>}
          <Button asChild variant="outline" className="mt-3">
            <Link href="/new">Start a new count</Link>
          </Button>
        </div>
      )}

      {job.status === "done" && <ResultPanel job={job} initialResult={initialResult} initialParsed={initialParsed} initialCorrections={initialCorrections} />}
    </div>
  );
}
