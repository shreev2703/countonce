import { NextResponse } from "next/server";

import { getSession } from "@/lib/server/session";

export const dynamic = "force-dynamic";

/** What the browser may know about the session (never the API key). */
export async function GET() {
  const s = await getSession();
  const body = s ? { signedIn: true, name: s.name, email: s.email } : { signedIn: false };
  return NextResponse.json(body, { headers: { "cache-control": "no-store" } });
}
