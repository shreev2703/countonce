import { NextResponse, type NextRequest } from "next/server";

import { safeNext } from "@/lib/safe-next";
import { devAuthEnabled } from "@/lib/server/config";
import { signInUser } from "@/lib/server/exchange";
import { cookieOptions, SESSION_COOKIE } from "@/lib/server/session";

/** Test-only sign-in (COUNTONCE_DEV_AUTH=1): lets Playwright and local runs skip Google. */
export async function POST(req: NextRequest) {
  if (!devAuthEnabled()) return NextResponse.json({ error: "not found" }, { status: 404 });
  const form = await req.formData().catch(() => null);
  const user = String(form?.get("user") ?? "tester").replace(/[^a-z0-9_-]/gi, "").slice(0, 32) || "tester";
  const token = await signInUser({ provider: "dev", subject: user, name: user, email: null });
  // A relative location keeps the browser's host, so the cookie follows the redirect.
  const res = new NextResponse(null, { status: 303, headers: { location: safeNext(form?.get("next")) } });
  res.cookies.set(SESSION_COOKIE, token, cookieOptions(req.nextUrl.protocol === "https:"));
  return res;
}
