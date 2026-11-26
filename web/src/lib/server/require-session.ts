import { redirect } from "next/navigation";

import { getSession } from "./session";

/** For pages that need a signed-in user: send everyone else to /signin and back. */
export async function requireSession(next: string) {
  const s = await getSession();
  if (!s) redirect(`/signin?next=${encodeURIComponent(next)}`);
  return s;
}
