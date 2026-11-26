import { apiUrl } from "./config";
import { parseResult, type Result } from "@/lib/schemas/result";

export interface SharedView {
  prompt: string;
  result: Result | null;
  finishedAt: number | null;
}

/** Read-only view for a share token, straight from the API (no sign-in); null if unknown/revoked. */
export async function fetchShare(token: string): Promise<SharedView | null> {
  const res = await fetch(`${apiUrl()}/share/${encodeURIComponent(token)}`, { cache: "no-store" }).catch(() => null);
  if (!res || !res.ok) return null;
  const body = (await res.json()) as { job: { prompt: string; finished_at: number | null }; result: { result_json: unknown } | null };
  let result: Result | null = null;
  try {
    result = body.result ? parseResult(body.result.result_json) : null;
  } catch {
    result = null;
  }
  return { prompt: body.job.prompt, result, finishedAt: body.job.finished_at };
}
