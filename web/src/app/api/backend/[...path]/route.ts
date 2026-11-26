import { NextResponse, type NextRequest } from "next/server";

import { apiUrl } from "@/lib/server/config";
import { getSession } from "@/lib/server/session";

export const dynamic = "force-dynamic";

/**
 * Same-origin proxy to the CountOnce API. It adds the signed-in user's API key from the session
 * cookie, so the browser never holds the key and no CORS is needed. Uploads and the SSE stream
 * pass through unbuffered. Only `GET /share/*`, `GET /health` and `GET /status` are open.
 */
const FORWARD_REQ = ["content-type", "idempotency-key", "x-chunk-sha256", "accept", "last-event-id"];
const FORWARD_RES = ["content-type", "cache-control", "retry-after", "content-disposition", "traceparent"];

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const rel = path.map(encodeURIComponent).join("/");
  const open = req.method === "GET" && (rel === "health" || rel === "status" || rel.startsWith("share/"));
  const headers = new Headers();
  for (const h of FORWARD_REQ) {
    const v = req.headers.get(h);
    if (v) headers.set(h, v);
  }
  if (!open) {
    const s = await getSession();
    if (!s) {
      return NextResponse.json(
        { error: { code: "ERR_UNAUTHENTICATED", message: "Sign in to continue", tip: null, retryable: false } },
        { status: 401 },
      );
    }
    headers.set("authorization", `Bearer ${s.apiKey}`);
  }
  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  let upstream: Response;
  try {
    upstream = await fetch(`${apiUrl()}/${rel}${req.nextUrl.search}`, {
      method: req.method,
      headers,
      body: hasBody ? req.body : undefined,
      // @ts-expect-error `duplex` is required by undici for streamed bodies
      duplex: hasBody ? "half" : undefined,
      redirect: "manual",
      cache: "no-store",
      signal: req.signal,
    });
  } catch {
    return NextResponse.json(
      { error: { code: "ERR_API_UNREACHABLE", message: "The CountOnce API is not reachable", tip: "Try again in a minute.", retryable: true } },
      { status: 502 },
    );
  }
  const out = new Headers();
  for (const h of FORWARD_RES) {
    const v = upstream.headers.get(h);
    if (v) out.set(h, v);
  }
  const loc = upstream.headers.get("location");
  if (loc && upstream.status >= 300 && upstream.status < 400) out.set("location", loc); // signed file URL
  if (out.get("content-type")?.startsWith("text/event-stream")) out.set("x-accel-buffering", "no");
  return new NextResponse(upstream.body, { status: upstream.status, headers: out });
}

export { proxy as GET, proxy as POST, proxy as PUT, proxy as DELETE };
