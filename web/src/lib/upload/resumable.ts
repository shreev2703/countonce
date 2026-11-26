/**
 * Resumable multipart upload against the API's chunked-upload endpoints (PLAN 14.8):
 * `POST /uploads` -> `PUT /uploads/{id}/chunks/{i}` (each with `X-Chunk-SHA256`) ->
 * `POST /uploads/{id}/finalize`. The session id is kept in storage, so after a reload or a
 * dropped connection `GET /uploads/{id}` says which chunks are already there and only the rest
 * are sent. Chunks are retried with backoff; 4xx other than 408/429 are not retried.
 */
import { sha256Blob, sha256Hex } from "./sha256";

export const DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024; // under the 4.5 MB serverless body cap

export interface UploadFile {
  name: string;
  size: number;
  lastModified: number;
  slice(start: number, end: number): Blob;
}

export interface KeyValueStore {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
  removeItem(k: string): void;
}

export type Fetcher = (path: string, init?: RequestInit) => Promise<Response>;

export interface UploadOptions {
  api: Fetcher;
  store: KeyValueStore;
  jobId: string;
  chunkBytes?: number;
  retries?: number;
  backoffMs?: number;
  signal?: AbortSignal;
  onProgress?: (sentBytes: number, totalBytes: number) => void;
  sleep?: (ms: number) => Promise<void>;
}

export class UploadFailure extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly retryable: boolean,
  ) {
    super(message);
    this.name = "UploadFailure";
  }
}

interface Saved {
  sessionId: string;
  chunkBytes: number;
  sha256: string;
  /** Finalized into the job: a retry of the whole submit must not upload it again. */
  done?: boolean;
}

export const sessionKey = (jobId: string, f: UploadFile) =>
  `co:upload:${jobId}:${f.name}:${f.size}:${f.lastModified}`;

async function fail(res: Response): Promise<never> {
  const body = (await res.json().catch(() => ({}))) as { error?: { code?: string; message?: string } };
  const retryable = res.status >= 500 || res.status === 408 || res.status === 429;
  throw new UploadFailure(body.error?.message ?? res.statusText, body.error?.code ?? "http_error", retryable);
}

async function withRetry<T>(o: UploadOptions, fn: () => Promise<T>): Promise<T> {
  const retries = o.retries ?? 4;
  const sleep = o.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (e) {
      if (o.signal?.aborted) throw e;
      const retryable = e instanceof UploadFailure ? e.retryable : true; // network errors
      if (!retryable || attempt >= retries) throw e;
      await sleep((o.backoffMs ?? 500) * 2 ** attempt);
    }
  }
}

/** Upload one file into the job. Resolves when the server has finalized it. */
export async function uploadFile(file: UploadFile, o: UploadOptions): Promise<{ resumed: boolean; skipped: number }> {
  const key = sessionKey(o.jobId, file);
  let saved: Saved | null = null;
  let have = new Set<number>();
  const raw = o.store.getItem(key);
  if (raw) {
    saved = JSON.parse(raw) as Saved;
    if (saved.done) {
      o.onProgress?.(file.size, file.size);
      return { resumed: true, skipped: Math.ceil(file.size / saved.chunkBytes) };
    }
    const res = await o.api(`/uploads/${saved.sessionId}`);
    if (res.ok) {
      const st = (await res.json()) as { received?: number[]; status?: string };
      if (st.status === "finalized") {
        o.store.setItem(key, JSON.stringify({ ...saved, done: true }));
        o.onProgress?.(file.size, file.size);
        return { resumed: true, skipped: Math.ceil(file.size / saved.chunkBytes) };
      }
      if (st.status === "open") have = new Set(st.received ?? []);
      else saved = null; // failed or purged session: start over
    } else {
      saved = null; // expired or unknown: start over
    }
    if (!saved) o.store.removeItem(key);
  }

  const chunkBytes = saved?.chunkBytes ?? o.chunkBytes ?? DEFAULT_CHUNK_BYTES;
  if (!saved) {
    const sha256 = await sha256Blob(file);
    const created = await o.api("/uploads", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ filename: file.name, total_bytes: file.size, sha256, chunk_size: chunkBytes }),
      signal: o.signal,
    });
    if (!created.ok) await fail(created);
    const body = (await created.json()) as { session_id: string };
    saved = { sessionId: body.session_id, chunkBytes, sha256 };
    o.store.setItem(key, JSON.stringify(saved));
  }

  const n = Math.max(1, Math.ceil(file.size / chunkBytes));
  let sent = 0;
  for (const i of have) sent += Math.min(chunkBytes, file.size - i * chunkBytes);
  o.onProgress?.(sent, file.size);
  for (let i = 0; i < n; i++) {
    if (have.has(i)) continue;
    const piece = new Uint8Array(await file.slice(i * chunkBytes, (i + 1) * chunkBytes).arrayBuffer());
    const digest = sha256Hex(piece);
    await withRetry(o, async () => {
      const res = await o.api(`/uploads/${saved!.sessionId}/chunks/${i}`, {
        method: "PUT",
        headers: { "content-type": "application/octet-stream", "x-chunk-sha256": digest },
        body: piece as BodyInit,
        signal: o.signal,
      });
      if (!res.ok) await fail(res);
    });
    sent += piece.length;
    o.onProgress?.(sent, file.size);
  }
  await withRetry(o, async () => {
    const res = await o.api(`/uploads/${saved!.sessionId}/finalize`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ job_id: o.jobId }),
      signal: o.signal,
    });
    if (!res.ok) await fail(res);
  });
  o.store.setItem(key, JSON.stringify({ ...saved, done: true }));
  return { resumed: have.size > 0, skipped: have.size };
}

/** Forget the upload bookkeeping of a job once it is submitted. */
export function clearUploadState(store: KeyValueStore & { length: number; key(i: number): string | null }, jobId: string): void {
  const prefix = `co:upload:${jobId}:`;
  const drop: string[] = [];
  for (let i = 0; i < store.length; i++) {
    const k = store.key(i);
    if (k?.startsWith(prefix)) drop.push(k);
  }
  drop.forEach((k) => store.removeItem(k));
}
