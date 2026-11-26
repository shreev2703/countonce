"use client";

import { useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";

export function Walkthrough({ stages }: { stages: { title: string; content: ReactNode }[] }) {
  const [i, setI] = useState(0);
  return (
    <div>
      <ol className="flex flex-wrap gap-2" aria-label="Pipeline stages">
        {stages.map((s, n) => (
          <li key={s.title}>
            <Button type="button" size="sm" variant={n === i ? "default" : "outline"} aria-current={n === i ? "step" : undefined} onClick={() => setI(n)}>
              {n + 1}. {s.title}
            </Button>
          </li>
        ))}
      </ol>
      <section aria-live="polite" className="mt-4 rounded-lg border border-border bg-card p-6">
        <h2 className="text-xl font-semibold">{stages[i]?.title}</h2>
        {stages[i]?.content}
      </section>
      <div className="mt-3 flex gap-2">
        <Button type="button" variant="outline" disabled={i === 0} onClick={() => setI(i - 1)}>
          Back
        </Button>
        <Button type="button" disabled={i === stages.length - 1} onClick={() => setI(i + 1)}>
          Next stage
        </Button>
      </div>
    </div>
  );
}
