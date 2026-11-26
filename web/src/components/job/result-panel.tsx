"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api, type JobView } from "@/lib/api/browser";
import { confidenceBadge, csvForResult, intervalText } from "@/lib/result-view/model";
import type { JobBundle } from "@/lib/server/job";
import type { Result } from "@/lib/schemas/result";
import { cn } from "@/lib/utils";

import { CorrectionsPanel } from "./corrections-panel";
import { FeedbackForm } from "./feedback-form";

const BADGE = {
  High: "border-solid text-status-new",
  Medium: "border-dashed text-status-duplicate",
  Low: "border-dotted text-status-rejected",
} as const;

export function ResultPanel({ job, initialResult, initialParsed, initialCorrections }: { job: JobView; initialResult?: JobBundle["result"]; initialParsed?: Result; initialCorrections?: JobBundle["corrections"] }) {
  const qc = useQueryClient();
  const seed = initialResult && initialParsed ? { raw: initialResult, result: initialParsed } : undefined;
  const q = useQuery({
    queryKey: ["result", job.id],
    queryFn: async () => {
      const raw = await api.getResult(job.id);
      // The schema library is only needed when the server could not hand over a parsed result.
      const { parseResult } = await import("@/lib/schemas/result");
      try {
        return { raw, result: parseResult(raw.result.result_json) };
      } catch (e) {
        throw new Error(`This result uses a format this page cannot read (${(e as Error).message})`);
      }
    },
    initialData: seed,
    staleTime: seed ? 60_000 : 0,
  });
  const corr = useQuery({ queryKey: ["corrections", job.id], queryFn: () => api.corrections(job.id), initialData: initialCorrections, staleTime: initialCorrections ? 60_000 : 0 });
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [shareErr, setShareErr] = useState(false);

  if (q.isError) return <p role="alert">{(q.error as Error).message.startsWith("This result") ? (q.error as Error).message : `Could not load the result: ${(q.error as Error).message}`}</p>;
  if (!q.data) return <p role="status">Loading result…</p>;

  const result = q.data.result;
  const conf = confidenceBadge(result);
  const corrected = corr.data?.corrected_count ?? null;

  const share = async () => {
    try {
      const s = await api.share(job.id);
      const url = `${window.location.origin}/s/${s.token}`;
      setShareUrl(url);
      await navigator.clipboard?.writeText(url).catch(() => undefined);
    } catch {
      setShareErr(true);
    }
  };
  const csv = () => {
    const url = URL.createObjectURL(new Blob([csvForResult(result)], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `count-${job.id.slice(0, 8)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-4">
      <Card className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between" aria-labelledby="count-heading">
        <div>
          <h2 id="count-heading" className="sr-only">
            Count result
          </h2>
          <p className="text-sm text-muted-foreground">{result.prompt}</p>
          <p data-testid="count" className="text-6xl font-semibold leading-none tabular-nums" aria-label={`Count: ${corrected ?? result.count} ${result.prompt}`}>
            {corrected ?? result.count}
          </p>
          {corrected !== null && corrected !== result.count && (
            <p className="mt-1 text-sm" data-testid="corrected-note">
              Your corrections changed this from {result.count}.
            </p>
          )}
          <p className="mt-2 text-sm" data-testid="interval">
            {intervalText(result)}
          </p>
          <p className="mt-2 text-sm">
            <span data-testid="confidence" className={cn("inline-flex rounded-full border-2 px-2.5 py-0.5 text-xs font-semibold", BADGE[conf.level])}>
              {conf.level} confidence
            </span>{" "}
            <span className="text-muted-foreground">because {conf.reason}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" onClick={csv}>
            CSV
          </Button>
          <Button type="button" variant="outline" onClick={() => void share()}>
            Create share link
          </Button>
        </div>
      </Card>
      {shareUrl && (
        <p data-testid="share-url" className="break-all rounded-md border border-border p-3 text-sm">
          Anyone with this link can see this result, not your photos: <a className="underline" href={shareUrl}>{shareUrl}</a>
        </p>
      )}
      {shareErr && <p role="alert">Could not create a share link. Try again.</p>}

      {result.warnings.length > 0 && (
        <Card aria-label="Warnings">
          <h3 className="font-semibold">Warnings</h3>
          <ul className="mt-2 list-disc pl-5 text-sm">
            {result.warnings.map((w) => (
              <li key={w.code}>
                {w.message}
                {w.fix ? <span className="text-muted-foreground"> {w.fix}</span> : null}
              </li>
            ))}
          </ul>
        </Card>
      )}

      <CorrectionsPanel
        jobId={job.id}
        result={result}
        onChanged={() => void qc.invalidateQueries({ queryKey: ["corrections", job.id] })}
      />
      <FeedbackForm jobId={job.id} count={corrected ?? result.count} />
    </div>
  );
}
