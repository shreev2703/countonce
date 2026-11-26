import { NextResponse, type NextRequest } from "next/server";

import { safeNext } from "@/lib/safe-next";
import { googleConfig, publicOrigin } from "@/lib/server/config";
import { checkIdToken, type IdToken } from "@/lib/server/google";
import { signInUser } from "@/lib/server/exchange";
import { cookieOptions, SESSION_COOKIE, unseal } from "@/lib/server/session";

export async function GET(req: NextRequest) {
  const fail = (code: string) => NextResponse.redirect(new URL(`/signin?error=${code}`, req.url));
  const cfg = googleConfig();
  const pending = unseal<{ state: string; verifier: string; next: string }>(req.cookies.get("co_oauth")?.value ?? "");
  const code = req.nextUrl.searchParams.get("code");
  if (!cfg || !pending || !code || req.nextUrl.searchParams.get("state") !== pending.state) return fail("oauth_state");

  const tok = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code,
      client_id: cfg.clientId,
      client_secret: cfg.clientSecret,
      redirect_uri: `${publicOrigin(req)}/api/auth/google/callback`,
      grant_type: "authorization_code",
      code_verifier: pending.verifier,
    }),
  });
  if (!tok.ok) return fail("oauth_exchange");
  const idToken = ((await tok.json()) as { id_token?: string }).id_token;
  if (!idToken) return fail("oauth_exchange");
  const claims = JSON.parse(Buffer.from(idToken.split(".")[1] ?? "", "base64url").toString("utf8")) as IdToken;
  if (!checkIdToken(claims, cfg.clientId, Date.now() / 1000)) return fail("oauth_claims");

  const token = await signInUser({
    provider: "google",
    subject: claims.sub,
    name: claims.name ?? claims.email ?? "Signed in",
    email: claims.email ?? null,
  });
  const next = safeNext(pending.next);
  const res = NextResponse.redirect(new URL(next, publicOrigin(req)));
  res.cookies.set(SESSION_COOKIE, token, cookieOptions(req.nextUrl.protocol === "https:"));
  res.cookies.delete("co_oauth");
  return res;
}
