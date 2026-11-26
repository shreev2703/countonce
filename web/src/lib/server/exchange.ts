import { apiUrl, authSecret } from "./config";
import { newSession } from "./session";

/** Trade a verified identity for an API key and return the sealed session cookie value. */
export async function signInUser(p: {
  provider: string;
  subject: string;
  name: string;
  email: string | null;
}): Promise<string> {
  const res = await fetch(`${apiUrl()}/auth/exchange`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-auth-secret": authSecret() },
    body: JSON.stringify({ provider: p.provider, subject: p.subject }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`API sign-in exchange failed: HTTP ${res.status}`);
  const out = (await res.json()) as { user_id: string; api_key: string };
  return newSession({ userId: out.user_id, apiKey: out.api_key, name: p.name, email: p.email });
}
