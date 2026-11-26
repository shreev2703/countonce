"use client";

import { m, useMotionTemplate, useMotionValue } from "framer-motion";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/** Card with a soft radial highlight that follows the pointer. Decorative only. */
export function SpotlightCard({ children, className }: { children: ReactNode; className?: string }) {
  const x = useMotionValue(-400);
  const y = useMotionValue(-400);
  const background = useMotionTemplate`radial-gradient(260px circle at ${x}px ${y}px, var(--spotlight), transparent 70%)`;
  return (
    <div
      className={cn("surface group relative h-full overflow-hidden rounded-xl", className)}
      onPointerMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        x.set(e.clientX - r.left);
        y.set(e.clientY - r.top);
      }}
    >
      <m.div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{ background }}
      />
      <div className="relative flex h-full flex-col p-5">{children}</div>
    </div>
  );
}
