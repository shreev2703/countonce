/**
 * Typed API client. Types come from the FastAPI OpenAPI schema (`npm run gen:api` ->
 * `schema.gen.ts`, input `web/openapi.json` from `scripts/export_openapi.py`).
 */
import type { components, paths } from "./schema.gen";

import { parseResult, type Result } from "@/lib/schemas/result";

export type { components, paths };

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly retryable: boolean,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    // Backend error envelope: {"error": {"code", "message", "tip", "retryable"}}
    const body = (await res.json().catch(() => ({}))) as {
      error?: { code?: string; message?: string; retryable?: boolean };
    };
    throw new ApiError(
      res.status,
      body.error?.code ?? "http_error",
      body.error?.message ?? res.statusText,
      body.error?.retryable ?? res.status >= 500,
    );
  }
  return res;
}

/** Fetch a result document and validate it by its `schema_version`. */
export async function fetchResult(url: string): Promise<Result> {
  const res = await apiFetch(url);
  return parseResult(await res.json());
}
