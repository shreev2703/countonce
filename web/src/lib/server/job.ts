import { apiUrl } from "./config";
import type { JobView } from "@/lib/api/browser";
import { parseResult, type Result } from "@/lib/schemas/result";

type Fetched<T> = T | undefined;

/** First-paint data for /jobs/[id]: the job, and for a finished job its result and corrections. */
export interface JobBundle {
  job?: JobView;
  result?: { job: JobView; result: { result_json: unknown; warnings_json: unknown }; outputs: { id: string; name: string; bytes: number; url: string }[] };
  /** `result_json` validated on the server, so the browser needs no schema library for first paint. */
  parsed?: Result;
  corrections?: { corrections: { action: string; detection_ids: string[] }[]; corrected_count: number | null };
}

async function get<T>(path: string, apiKey: string): Promise<Fetched<T>> {
  const res = await fetch(`${apiUrl()}${path}`, { cache: "no-store", headers: { authorization: `Bearer ${apiKey}` } }).catch(() => null);
  return res?.ok ? ((await res.json()) as T) : undefined;
}

/** Best effort: anything that fails here is simply fetched again by the browser. */
export async function fetchJobBundle(id: string, apiKey: string): Promise<JobBundle> {
  const job = await get<JobView>(`/jobs/${encodeURIComponent(id)}`, apiKey);
  if (job?.status !== "done") return { job };
  const [result, corrections] = await Promise.all([
    get<NonNullable<JobBundle["result"]>>(`/jobs/${encodeURIComponent(id)}/result`, apiKey),
    get<NonNullable<JobBundle["corrections"]>>(`/jobs/${encodeURIComponent(id)}/corrections`, apiKey),
  ]);
  let parsed: Result | undefined;
  try {
    if (result) parsed = parseResult(result.result.result_json);
  } catch {
    // Unreadable format: the browser re-parses and shows the error message.
  }
  return { job, result, parsed, corrections };
}
