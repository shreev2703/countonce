"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

interface Me {
  signedIn: boolean;
  name?: string;
}

/** Header links that depend on the session. Fetched on the client so `/` and `/demo` stay static. */
export function AuthNav() {
  const q = useQuery({
    queryKey: ["me"],
    queryFn: async () => (await fetch("/api/auth/me")).json() as Promise<Me>,
    staleTime: 60_000,
  });
  const link = "underline-offset-4 hover:underline";
  if (!q.data?.signedIn) {
    return (
      <Link href="/signin" className={link}>
        Sign in
      </Link>
    );
  }
  return (
    <>
      <Link href="/new" className={link}>
        New count
      </Link>
      <Link href="/history" className={link}>
        History
      </Link>
      <form action="/api/auth/signout" method="post">
        <button type="submit" className={link} aria-label={`Sign out ${q.data.name ?? ""}`.trim()}>
          Sign out
        </button>
      </form>
    </>
  );
}
