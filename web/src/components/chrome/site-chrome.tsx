import Link from "next/link";
import type { ReactNode } from "react";

export function SiteHeader() {
  return (
    <header className="border-b border-border">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <Link href="/" className="text-lg font-semibold">
          CountOnce
        </Link>
        <nav aria-label="Main" className="flex gap-5 text-sm">
          <Link href="/demo" className="underline-offset-4 hover:underline">
            Demo gallery
          </Link>
        </nav>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-border">
      <div className="mx-auto max-w-5xl px-4 py-8 text-sm text-muted-foreground">
        CountOnce is a research project. Demo scenes are precomputed and labelled as such; numbers
        shown on this site come from real runs.
      </div>
    </footer>
  );
}

export function PageShell({ children }: { children: ReactNode }) {
  return (
    <>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded-md focus:bg-card focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <SiteHeader />
      <main id="main">{children}</main>
      <SiteFooter />
    </>
  );
}
