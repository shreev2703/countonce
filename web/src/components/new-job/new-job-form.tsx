"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api/browser";
import { bff } from "@/lib/api/browser";
import { canSubmit, probeImage, runPreflight, type Check, type FileInfo } from "@/lib/upload/preflight";
import { clearUploadState, uploadFile } from "@/lib/upload/resumable";
import { cn } from "@/lib/utils";

import { CaptureGuide } from "./capture-guide";
import { ExampleBox, type Box } from "./example-box";

const LEVEL_STYLE = { pass: "text-status-new", warn: "text-status-duplicate", fail: "text-status-rejected" } as const;
const LEVEL_MARK = { pass: "OK", warn: "Check", fail: "Fix" } as const;
const ATTEMPT_KEY = "co:new:attempt";

interface Attempt {
  idempotencyKey: string;
  jobId?: string;
}

function loadAttempt(): Attempt {
  const raw = sessionStorage.getItem(ATTEMPT_KEY);
  if (raw) return JSON.parse(raw) as Attempt;
  const a = { idempotencyKey: crypto.randomUUID() };
  sessionStorage.setItem(ATTEMPT_KEY, JSON.stringify(a));
  return a;
}

export function NewJobForm() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [infos, setInfos] = useState<FileInfo[]>([]);
  const [online, setOnline] = useState(true);
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [box, setBox] = useState<Box | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ message: string; tip?: string } | null>(null);
  const [progress, setProgress] = useState<Record<string, number>>({});
  const submitting = useRef(false);
  const lastFiles = useRef<File[]>([]);

  useEffect(() => {
    const on = () => setOnline(navigator.onLine);
    on();
    window.addEventListener("online", on);
    window.addEventListener("offline", on);
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", on);
    };
  }, []);

  useEffect(() => {
    let live = true;
    void Promise.all(files.map(probeImage)).then((r) => live && setInfos(r));
    return () => {
      live = false;
    };
  }, [files]);

  const firstImage = useMemo(() => files.find((f) => f.type.startsWith("image/")), [files]);
  const previewUrl = useMemo(() => (firstImage ? URL.createObjectURL(firstImage) : null), [firstImage]);
  useEffect(() => () => void (previewUrl && URL.revokeObjectURL(previewUrl)), [previewUrl]);

  const checks: Check[] = useMemo(
    () =>
      runPreflight(files.length === infos.length ? infos : files.map((f) => ({ name: f.name, size: f.size })), {
        online,
        hasCrypto: typeof crypto !== "undefined" && !!crypto.randomUUID,
        hasSlice: typeof Blob !== "undefined" && "slice" in Blob.prototype,
      }),
    [files, infos, online],
  );
  const ready = canSubmit(checks) && prompt.trim().length > 0 && !busy;

  const add = (list: FileList | null) => {
    if (!list) return;
    setFiles((cur) => [...cur, ...Array.from(list)]);
  };

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (submitting.current || !ready) return; // a double tap must not create two jobs
    submitting.current = true;
    setBusy(true);
    setError(null);
    try {
      const attempt = loadAttempt();
      const job = await api.createJob(
        {
          prompt: prompt.trim(),
          negative_prompts: negative.split("\n").map((s) => s.trim()).filter(Boolean),
          options: box ? { example_box: { file: firstImage?.name, ...box } } : {},
        },
        attempt.idempotencyKey,
      );
      sessionStorage.setItem(ATTEMPT_KEY, JSON.stringify({ ...attempt, jobId: job.id }));
      lastFiles.current = files;
      for (const f of files) {
        await uploadFile(f, {
          api: bff,
          store: localStorage,
          jobId: job.id,
          onProgress: (sent, total) => setProgress((p) => ({ ...p, [f.name + f.size]: sent / total })),
        });
      }
      await api.submit(job.id, attempt.idempotencyKey);
      clearUploadState(localStorage, job.id);
      sessionStorage.removeItem(ATTEMPT_KEY);
      router.push(`/jobs/${job.id}`);
    } catch (err) {
      const e2 = err as Error & { tip?: string };
      setError({ message: e2.message, tip: e2.tip });
      submitting.current = false;
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-6" aria-label="New count">
      <CaptureGuide />

      <Card className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold">1. Add your photos</h2>
        <div className="flex flex-wrap gap-2">
          <label className="inline-flex h-11 cursor-pointer items-center rounded-md border border-border bg-card px-4 text-sm font-medium hover:bg-muted focus-within:ring-2 focus-within:ring-ring">
            Choose photos or video
            <input type="file" multiple accept="image/jpeg,image/png,image/webp,image/bmp,video/*" className="sr-only" data-testid="file-input" onChange={(e) => add(e.target.files)} />
          </label>
          <label className="inline-flex h-11 cursor-pointer items-center rounded-md border border-border bg-card px-4 text-sm font-medium hover:bg-muted focus-within:ring-2 focus-within:ring-ring">
            Take a photo
            <input type="file" accept="image/*" capture="environment" className="sr-only" data-testid="camera-input" onChange={(e) => add(e.target.files)} />
          </label>
          {files.length > 0 && (
            <Button type="button" variant="ghost" onClick={() => setFiles([])}>
              Clear all
            </Button>
          )}
        </div>
        <p aria-live="polite" className="text-sm text-muted-foreground">
          {files.length} file{files.length === 1 ? "" : "s"} selected
        </p>
        <ul aria-label="Pre-flight checks" className="flex flex-col gap-1.5" data-testid="preflight">
          {checks.map((c) => (
            <li key={c.id} data-level={c.level} className="text-sm">
              <span className={cn("mr-2 inline-block w-12 font-semibold", LEVEL_STYLE[c.level])}>{LEVEL_MARK[c.level]}</span>
              <span className="font-medium">{c.title}.</span> <span className="text-muted-foreground">{c.detail}</span>
            </li>
          ))}
        </ul>
      </Card>

      <Card className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold">2. What should we count?</h2>
        <label className="flex flex-col gap-1 text-sm font-medium">
          Count these
          <input required value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="e.g. red apples" className="h-11 rounded-md border border-border bg-background px-3 text-base font-normal" data-testid="prompt" />
        </label>
        <label className="flex flex-col gap-1 text-sm font-medium">
          Do not count (optional, one per line)
          <textarea value={negative} onChange={(e) => setNegative(e.target.value)} rows={2} placeholder="e.g. green apples" className="rounded-md border border-border bg-background px-3 py-2 text-base font-normal" data-testid="negative" />
        </label>
        {previewUrl && (
          <div>
            <p className="mb-1 text-sm font-medium">Show one example (optional)</p>
            <ExampleBox src={previewUrl} value={box} onChange={setBox} />
          </div>
        )}
      </Card>

      {error && (
        <div role="alert" className="rounded-md border border-status-rejected p-3 text-sm" data-testid="submit-error">
          <p className="font-medium">{error.message}</p>
          {error.tip && <p className="text-muted-foreground">{error.tip}</p>}
          <p className="mt-1 text-muted-foreground">Nothing is lost: press the button again and the upload continues where it stopped.</p>
        </div>
      )}
      {busy && (
        <div role="status" className="text-sm" data-testid="upload-progress">
          Uploading…{" "}
          {files.map((f) => (
            <span key={f.name + f.size} className="mr-2">
              {Math.round((progress[f.name + f.size] ?? 0) * 100)}%
            </span>
          ))}
        </div>
      )}
      <Button type="submit" size="lg" disabled={!ready} data-testid="submit" className="self-start">
        {busy ? "Working…" : "Count"}
      </Button>
    </form>
  );
}
