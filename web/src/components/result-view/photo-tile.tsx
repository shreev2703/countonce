"use client";

import { objectLabel, type CameraPose, type Detection } from "@/lib/result-view/model";
import { cn } from "@/lib/utils";

/** Side of the drawn box in image pixels. The export keeps only the detection centre (uv). */
export const BOX_PX = 18;

const DASH: Record<Detection["status"], string | undefined> = {
  new: undefined,
  duplicate: "5 3",
  rejected: "1.5 3",
};
const STROKE: Record<Detection["status"], string> = {
  new: "var(--status-new)",
  duplicate: "var(--status-duplicate)",
  rejected: "var(--status-rejected)",
};

export function PhotoTile({
  base,
  frame,
  cam,
  detections,
  highlightObject,
  onSelectObject,
  onHoverObject,
  interactive = true,
  className,
}: {
  base: string;
  frame: string;
  cam: CameraPose;
  detections: Detection[];
  highlightObject: number | null;
  onSelectObject: (id: number | null) => void;
  onHoverObject: (id: number | null) => void;
  /** false inside another button (photo strip): boxes are drawn but not focusable or clickable. */
  interactive?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("relative overflow-hidden rounded-md border border-border bg-muted", className)}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`${base}/thumbs/${frame}.jpg`}
        alt={`Photo ${cam.view + 1}`}
        className="block w-full [image-rendering:pixelated]"
        style={{ aspectRatio: `${cam.width} / ${cam.height}` }}
      />
      <svg viewBox={`0 0 ${cam.width} ${cam.height}`} className="absolute inset-0 h-full w-full" aria-hidden={interactive ? undefined : true}>
        {detections
          .filter((d) => d.uv)
          .map((d) => {
            const [u, v] = d.uv as [number, number];
            const hot = d.objectId !== null && d.objectId === highlightObject;
            const label =
              d.status === "rejected"
                ? "discarded sighting"
                : `object ${objectLabel(d.objectId ?? 0)}, ${d.status === "new" ? "first sighting" : "seen again"}`;
            const select = () => onSelectObject(d.objectId);
            return (
              <g
                key={d.id}
                role={interactive ? "button" : undefined}
                tabIndex={interactive ? 0 : undefined}
                aria-label={interactive ? `Photo ${cam.view + 1}: ${label}` : undefined}
                data-testid={interactive ? `box-${d.id}` : undefined}
                data-status={d.status}
                data-hot={hot}
                opacity={highlightObject !== null && !hot ? 0.35 : 1}
                className="cursor-pointer outline-none focus-visible:[&>rect]:stroke-[3]"
                onClick={select}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    select();
                  }
                }}
                onPointerEnter={() => onHoverObject(d.objectId)}
                onPointerLeave={() => onHoverObject(null)}
                onFocus={() => onHoverObject(d.objectId)}
                onBlur={() => onHoverObject(null)}
              >
                <rect
                  x={u - BOX_PX / 2}
                  y={v - BOX_PX / 2}
                  width={BOX_PX}
                  height={BOX_PX}
                  fill={hot ? STROKE[d.status] : "transparent"}
                  fillOpacity={hot ? 0.3 : 0}
                  stroke={STROKE[d.status]}
                  strokeWidth={hot ? 3 : 1.5}
                  strokeDasharray={DASH[d.status]}
                />
                {d.objectId !== null && (
                  <text pointerEvents="none" x={u - BOX_PX / 2} y={v - BOX_PX / 2 - 2} fontSize="9" fontWeight="700" fill="#fff" stroke="#000" strokeWidth="2.5" paintOrder="stroke">
                    {objectLabel(d.objectId)}
                  </text>
                )}
              </g>
            );
          })}
      </svg>
    </div>
  );
}
