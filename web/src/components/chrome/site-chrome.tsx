import Link from "next/link";
import type { ReactNode } from "react";

import { AuthNav } from "./auth-nav";

/** `auth={false}` keeps a page free of session requests (the demo works with no server APIs). */
export function SiteHeader({ auth = true }: { auth?: boolean }) {
  return (
    <header className="border-b border-border">
      <div className="mx-auto flex min-h-14 max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-2">
        <Link href="/" className="text-lg font-semibold">
          CountOnce
        </Link>
        <nav aria-label="Main" className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
          <Link href="/demo" className="underline-offset-4 hover:underline">
            Demo gallery
          </Link>
          <Link href="/how-it-works" className="underline-offset-4 hover:underline">
            How it works
          </Link>
          <Link href="/benchmark" className="underline-offset-4 hover:underline">
            Benchmark
          </Link>
          {auth && <AuthNav />}
        </nav>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-border">
      <div className="mx-auto max-w-6xl px-4 py-8 text-sm text-muted-foreground">
        CountOnce is a research project. Demo scenes are precomputed and labelled as such; numbers
        shown on this site come from real runs.
        <nav aria-label="Footer" className="mt-3 flex flex-wrap gap-4">
          {[
            ["/status", "Status"],
            ["/about", "About"],
            ["/privacy", "Privacy"],
          ].map(([href, label]) => (
            <Link key={href} href={href!} className="underline underline-offset-4">
              {label}
            </Link>
          ))}
        </nav>
      </div>
    </footer>
  );
}

export function PageShell({
  children,
  obsidian = false,
  auth = true,
}: {
  children: ReactNode;
  obsidian?: boolean;
  auth?: boolean;
}) {
  return (
    <div className={obsidian ? "obsidian" : undefined}>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded-md focus:bg-card focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <SiteHeader auth={auth} />
      <main id="main">{children}</main>
      <SiteFooter />
    </div>
  );
}
