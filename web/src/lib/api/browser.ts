/** Browser-side API helpers. Everything goes through the same-origin BFF proxy at /api/backend. */
import { ApiError } from "./client";
import type { components } from "./schema.gen";

export const BFF = "/api/backend";

export type JobView = {
  id: string;
  prompt: string;
  status: string;
  stage: string | null;
  progress: number;
  error_code: string | null;
  error_message: string | null;
  tip?: string;
  attempts: number;
  created_at: number;
  finished_at: number | null;
  negative_prompts: string[];
  options: Record<string, unknown>;
  files?: number;
};
export type Health = { status: string; workers_online: number; processing_paused: boolean; queue: { queued: number; running: number } };
export type PublicStatus = Omit<Health, "status"> & {
  recent: { status: string; finished_at: number; wait_s: number | null; run_s: number | null }[];
};
export type { components };

export async function bff(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${BFF}${path}`, init);
  if (res.status === 401 && typeof window !== "undefined") {
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination -- full reload clears client state
    window.location.href = `/signin?next=${encodeURIComponent(window.location.pathname)}`;
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: { code?: string; message?: string; tip?: string; retryable?: boolean } };
    const e = new ApiError(res.status, body.error?.code ?? "http_error", body.error?.message ?? res.statusText, body.error?.retryable ?? res.status >= 500);
    (e as ApiError & { tip?: string }).tip = body.error?.tip;
    throw e;
  }
  return res;
}

const json = async <T>(p: string, init?: RequestInit) => (await bff(p, init)).json() as Promise<T>;
const post = (body: unknown, headers: Record<string, string> = {}): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json", ...headers },
  body: JSON.stringify(body),
});

export const api = {
  health: () => json<Health>("/health"),
  status: () => json<PublicStatus>("/status"),
  listJobs: () => json<{ jobs: JobView[] }>("/jobs"),
  getJob: (id: string) => json<JobView>(`/jobs/${id}`),
  createJob: (body: { prompt: string; negative_prompts: string[]; options: Record<string, unknown> }, idempotencyKey: string) =>
    json<{ id: string; status: string; idempotent_replay: boolean }>("/jobs", post(body, { "idempotency-key": idempotencyKey })),
  submit: (id: string, idempotencyKey: string) =>
    json<JobView & { idempotent_replay: boolean }>(`/jobs/${id}/submit`, post({}, { "idempotency-key": idempotencyKey })),
  deleteJob: (id: string) => json<{ id: string }>(`/jobs/${id}`, { method: "DELETE" }),
  getResult: (id: string) => json<{ job: JobView; result: { result_json: unknown; warnings_json: unknown }; outputs: { id: string; name: string; bytes: number; url: string }[] }>(`/jobs/${id}/result`),
  share: (id: string) => json<{ token: string; url: string }>(`/jobs/${id}/share`, post({})),
  feedback: (id: string, body: { correct: boolean; user_count?: number | null; comment?: string | null }) =>
    json<{ id: string }>(`/jobs/${id}/feedback`, post(body)),
  corrections: (id: string) => json<{ corrections: { action: string; detection_ids: string[] }[]; corrected_count: number | null }>(`/jobs/${id}/corrections`),
  correct: (id: string, body: { action: "reject" | "merge" | "split"; detection_ids: string[]; consent?: boolean }) =>
    json<{ corrected_count: number; corrections: number; idempotent_replay: boolean }>(`/jobs/${id}/corrections`, post(body)),
};

export const TERMINAL = ["done", "failed", "dead_letter", "deleted"];
