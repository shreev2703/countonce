"use client";

import { m } from "framer-motion";
import { useState } from "react";

import { useMedia } from "@/lib/use-media";

/**
 * Illustration (not a result): three overlapping photos see 9 object sightings; each sighting flies
 * to its place in one shared scene, where sightings of the same object merge into 5 objects.
 * Plays once and can be replayed, so nothing moves indefinitely.
 */
const COLORS = ["#7aa2ff", "#5eead4", "#c4a1ff", "#fbbf77", "#f9a8d4"];
const PHOTOS = [
  { x: 8, label: "Photo 1", sees: [0, 1, 2] },
  { x: 196, label: "Photo 2", sees: [1, 2, 3] },
  { x: 384, label: "Photo 3", sees: [2, 3, 4] },
];
const TARGETS = [
  [112, 304],
  [194, 336],
  [280, 296],
  [366, 336],
  [448, 304],
] as const;
const PHOTO_W = 168;
const SIGHTINGS = PHOTOS.flatMap((p, pi) =>
  p.sees.map((obj, i) => ({
    key: `${pi}-${obj}`,
    obj,
    x: p.x + 38 + i * 46,
    y: 62 + (obj % 2) * 26,
    delay: 0.5 + pi * 0.35 + i * 0.08,
  })),
);

export function HeroDiagram() {
  const reduced = useMedia("(prefers-reduced-motion: reduce)");
  const [run, setRun] = useState(0);
  const t = (duration: number, delay: number) => (reduced ? { duration: 0 } : { duration, delay, ease: [0.22, 1, 0.36, 1] as const });

  return (
    <figure className="surface relative rounded-2xl p-3">
      <svg
        key={run}
        viewBox="0 0 560 400"
        className="h-auto w-full"
        role="img"
        aria-label="Illustration: three overlapping photos contain 9 sightings, which merge into 5 distinct objects in one 3D scene."
      >
        <defs>
          <radialGradient id="floor" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#4f6bd8" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#4f6bd8" stopOpacity="0" />
          </radialGradient>
        </defs>

        {PHOTOS.map((p, pi) => (
          <m.g key={p.label} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={t(0.5, pi * 0.12)}>
            <rect x={p.x} y="14" width={PHOTO_W} height="124" rx="10" fill="#ffffff" fillOpacity="0.035" stroke="#ffffff" strokeOpacity="0.14" />
            <text x={p.x + 12} y="34" fontSize="12" fill="#a1a8b8">
              {p.label}
            </text>
            <path d={`M${p.x + PHOTO_W / 2} 138 L280 250`} stroke="#ffffff" strokeOpacity="0.12" strokeDasharray="3 5" fill="none" />
          </m.g>
        ))}

        <ellipse cx="280" cy="316" rx="260" ry="78" fill="url(#floor)" />
        {[0, 1, 2, 3, 4].map((i) => (
          <path key={`h${i}`} d={`M${110 - i * 22} ${276 + i * 20} H${450 + i * 22}`} stroke="#ffffff" strokeOpacity="0.07" />
        ))}
        {[-3, -2, -1, 0, 1, 2, 3].map((i) => (
          <path key={`v${i}`} d={`M${280 + i * 55} 276 L${280 + i * 85} 356`} stroke="#ffffff" strokeOpacity="0.07" />
        ))}
        <text x="280" y="262" fontSize="12" fill="#a1a8b8" textAnchor="middle">
          One shared 3D scene
        </text>

        {SIGHTINGS.map((s) => (
          <circle key={`ghost-${s.key}`} cx={s.x} cy={s.y} r="11" fill={COLORS[s.obj]} fillOpacity="0.22" stroke={COLORS[s.obj]} />
        ))}
        {SIGHTINGS.map((s) => {
          const [tx, ty] = TARGETS[s.obj]!;
          return (
            <m.circle
              key={s.key}
              r="11"
              fill={COLORS[s.obj]}
              initial={{ cx: s.x, cy: s.y, opacity: 0 }}
              animate={{ cx: tx, cy: ty, opacity: 1 }}
              transition={t(1.1, s.delay)}
            />
          );
        })}
        {TARGETS.map(([x, y], i) => (
          <m.g key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={t(0.4, 2.5 + i * 0.1)}>
            <circle cx={x} cy={y} r="17" fill="none" stroke={COLORS[i]} strokeOpacity="0.5" />
            <text x={x} y={y + 4} fontSize="11" fontWeight="700" textAnchor="middle" fill="#0a0c11">
              {i + 1}
            </text>
          </m.g>
        ))}

        <text x="280" y="388" fontSize="17" textAnchor="middle" fill="#a1a8b8">
          9 sightings →{" "}
          <tspan fontWeight="700" fill="#f4f6fb">
            5 objects
          </tspan>
        </text>
      </svg>
      <figcaption className="flex items-center justify-between gap-3 px-2 pb-1 text-xs text-muted-foreground">
        <span>Illustration of the idea, not a measured result.</span>
        <button type="button" className="glass-btn" onClick={() => setRun((n) => n + 1)}>
          Replay
        </button>
      </figcaption>
    </figure>
  );
}
