import { createHash, randomBytes } from "node:crypto";

import { NextResponse, type NextRequest } from "next/server";

import { googleConfig, publicOrigin } from "@/lib/server/config";
import { cookieOptions, seal } from "@/lib/server/session";

export const OAUTH_COOKIE = "co_oauth";

/** Start Google sign-in: authorization code flow with state and PKCE. */
export async function GET(req: NextRequest) {
  const cfg = googleConfig();
  if (!cfg) return NextResponse.redirect(new URL("/signin?error=google_not_configured", req.url));
  const state = randomBytes(16).toString("base64url");
  const verifier = randomBytes(32).toString("base64url");
  const next = req.nextUrl.searchParams.get("next") ?? "/new";
  const url = new URL("https://accounts.google.com/o/oauth2/v2/auth");
  url.search = new URLSearchParams({
    client_id: cfg.clientId,
    redirect_uri: `${publicOrigin(req)}/api/auth/google/callback`,
    response_type: "code",
    scope: "openid email profile",
    state,
    code_challenge: createHash("sha256").update(verifier).digest("base64url"),
    code_challenge_method: "S256",
  }).toString();
  const res = NextResponse.redirect(url);
  res.cookies.set(OAUTH_COOKIE, seal({ state, verifier, next }), cookieOptions(req.nextUrl.protocol === "https:", 600));
  return res;
}
