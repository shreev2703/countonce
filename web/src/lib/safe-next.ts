const BASE = "http://internal.invalid";

/**
 * A same-site path to return to after sign-in, or `fallback`. A prefix check is not enough:
 * URL parsing treats `\` as `/` and drops tabs and newlines, so `/\host` leaves the site.
 */
export function safeNext(v: unknown, fallback = "/new"): string {
  if (typeof v !== "string" || !v.startsWith("/")) return fallback;
  const u = new URL(v, BASE);
  const path = u.pathname + u.search + u.hash;
  return u.origin === BASE && !path.startsWith("//") ? path : fallback;
}
