import { createCipheriv, createDecipheriv, createHash, randomBytes } from "node:crypto";

import { cookies } from "next/headers";

import { sessionSecret } from "./config";

export const SESSION_COOKIE = "co_session";
const MAX_AGE_S = 60 * 60 * 24 * 14;

export interface Session {
  userId: string;
  name: string;
  email: string | null;
  /** The API key for this user; lives only in the encrypted cookie and the BFF proxy. */
  apiKey: string;
  exp: number;
}

function key(): Buffer {
  return createHash("sha256").update(sessionSecret()).digest();
}

/** AES-256-GCM: the cookie is both unreadable and tamper-evident. */
export function seal(value: unknown): string {
  const iv = randomBytes(12);
  const c = createCipheriv("aes-256-gcm", key(), iv);
  const body = Buffer.concat([c.update(JSON.stringify(value), "utf8"), c.final()]);
  return Buffer.concat([iv, c.getAuthTag(), body]).toString("base64url");
}

export function unseal<T>(token: string): T | null {
  try {
    const raw = Buffer.from(token, "base64url");
    if (raw.length < 29) return null;
    const d = createDecipheriv("aes-256-gcm", key(), raw.subarray(0, 12));
    d.setAuthTag(raw.subarray(12, 28));
    const out = Buffer.concat([d.update(raw.subarray(28)), d.final()]);
    return JSON.parse(out.toString("utf8")) as T;
  } catch {
    return null;
  }
}

export function cookieOptions(secure: boolean, maxAge = MAX_AGE_S) {
  return { httpOnly: true, sameSite: "lax" as const, secure, path: "/", maxAge };
}

export function readSession(token: string | undefined): Session | null {
  if (!token) return null;
  const s = unseal<Session>(token);
  return s && s.exp > Date.now() / 1000 ? s : null;
}

export async function getSession(): Promise<Session | null> {
  return readSession((await cookies()).get(SESSION_COOKIE)?.value);
}

export function newSession(p: Omit<Session, "exp">): string {
  return seal({ ...p, exp: Math.floor(Date.now() / 1000) + MAX_AGE_S });
}
