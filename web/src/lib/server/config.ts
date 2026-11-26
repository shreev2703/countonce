/** Server-only configuration, all from environment variables (see web/README.md). */

export function apiUrl(): string {
  return (process.env.COUNTONCE_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

/** Shared secret with the API's `/auth/exchange` (`COUNTONCE_AUTH_SECRET` on the API side). */
export function authSecret(): string {
  const s = process.env.COUNTONCE_AUTH_SECRET;
  if (!s) throw new Error("COUNTONCE_AUTH_SECRET is not set");
  return s;
}

export function sessionSecret(): string {
  const s = process.env.COUNTONCE_SESSION_SECRET;
  if (!s || s.length < 16) throw new Error("COUNTONCE_SESSION_SECRET must be set (16+ characters)");
  return s;
}

export function googleConfig(): { clientId: string; clientSecret: string } | null {
  const clientId = process.env.GOOGLE_CLIENT_ID;
  const clientSecret = process.env.GOOGLE_CLIENT_SECRET;
  return clientId && clientSecret ? { clientId, clientSecret } : null;
}

/** The no-Google test sign-in. Never enable it on a public deployment. */
export function devAuthEnabled(): boolean {
  return process.env.COUNTONCE_DEV_AUTH === "1";
}

/** Public origin used for OAuth redirects (behind a proxy `request.url` is internal). */
export function publicOrigin(req: Request): string {
  return (process.env.COUNTONCE_PUBLIC_URL ?? new URL(req.url).origin).replace(/\/$/, "");
}
