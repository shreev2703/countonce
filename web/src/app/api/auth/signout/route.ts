import { NextResponse } from "next/server";

import { SESSION_COOKIE } from "@/lib/server/session";

export async function POST() {
  const res = new NextResponse(null, { status: 303, headers: { location: "/" } });
  res.cookies.delete(SESSION_COOKIE);
  return res;
}
