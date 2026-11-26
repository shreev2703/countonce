"use client";

import { useRef, useState } from "react";

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Draw one box around an example object on the first photo; coordinates are 0-1 fractions. */
export function ExampleBox({ src, value, onChange }: { src: string; value: Box | null; onChange: (b: Box | null) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const start = useRef<{ x: number; y: number } | null>(null);
  const [draft, setDraft] = useState<Box | null>(null);

  const pt = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  };
  const norm = (a: { x: number; y: number }, b: { x: number; y: number }): Box => ({
    x0: Math.min(a.x, b.x),
    y0: Math.min(a.y, b.y),
    x1: Math.max(a.x, b.x),
    y1: Math.max(a.y, b.y),
  });
  const shown = draft ?? value;

  return (
    <div className="flex flex-col gap-2">
      <div
        ref={ref}
        data-testid="example-box-canvas"
        className="relative w-full max-w-md touch-none select-none overflow-hidden rounded-md border border-border"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          start.current = pt(e);
        }}
        onPointerMove={(e) => {
          if (start.current) setDraft(norm(start.current, pt(e)));
        }}
        onPointerUp={(e) => {
          if (start.current) {
            const b = norm(start.current, pt(e));
            onChange(b.x1 - b.x0 > 0.02 && b.y1 - b.y0 > 0.02 ? b : null);
          }
          start.current = null;
          setDraft(null);
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt="First photo; drag to draw a box around one example object" className="block w-full" draggable={false} />
        {shown && (
          <div
            data-testid="example-box"
            className="pointer-events-none absolute border-2 border-accent bg-accent/20"
            style={{ left: `${shown.x0 * 100}%`, top: `${shown.y0 * 100}%`, width: `${(shown.x1 - shown.x0) * 100}%`, height: `${(shown.y1 - shown.y0) * 100}%` }}
          />
        )}
      </div>
      <button type="button" className="self-start text-sm underline underline-offset-4 disabled:opacity-40" disabled={!value} onClick={() => onChange(null)}>
        Clear box
      </button>
    </div>
  );
}
