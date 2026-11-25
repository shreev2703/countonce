"use client";

import { useMemo, useState } from "react";

import { StatusBadge } from "@/components/ui/status-badge";
import type { DemoScene } from "@/lib/result-view/demo";
import {
  explainPair,
  objectLabel,
  statusLabel,
  type CameraPose,
  type Detection,
  type Status,
} from "@/lib/result-view/model";
import { cn } from "@/lib/utils";

import { PhotoTile } from "./photo-tile";

export interface PanelCtx {
  scene: DemoScene;
  cameras: CameraPose[];
  detections: Detection[];
  selectedObject: number | null;
  highlightObject: number | null;
  onSelectObject: (id: number | null) => void;
  onHoverObject: (id: number | null) => void;
}

export function PhotosTab({ ctx }: { ctx: PanelCtx }) {
  const [filter, setFilter] = useState<Status | "all">("all");
  const filters: (Status | "all")[] = ["all", "new", "duplicate", "rejected"];
  return (
    <div>
      <div role="group" aria-label="Filter detections by status" className="mb-3 flex flex-wrap gap-2">
        {filters.map((f) => (
          <button
            key={f}
            type="button"
            aria-pressed={filter === f}
            onClick={() => setFilter(f)}
            className={cn(
              "rounded-full border border-border px-3 py-1 text-sm capitalize",
              filter === f && "bg-accent text-accent-foreground",
            )}
          >
            {f}
          </button>
        ))}
      </div>
      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {ctx.cameras.map((cam) => {
          const frame = ctx.scene.manifest.frames[cam.view] ?? "";
          const dets = ctx.detections.filter((d) => d.view === cam.view && (filter === "all" || d.status === filter));
          return (
            <li key={cam.view} data-testid={`photo-${cam.view}`}>
              <PhotoTile
                base={ctx.scene.base}
                frame={frame}
                cam={cam}
                detections={dets}
                highlightObject={ctx.highlightObject}
                onSelectObject={ctx.onSelectObject}
                onHoverObject={ctx.onHoverObject}
              />
              <p className="mt-1 text-sm text-muted-foreground">
                Photo {cam.view + 1}: {dets.length} shown
              </p>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

const fmt = (v: number | null, d = 2) => (v === null ? "n/a" : v.toFixed(d));

export function ExplainTab({ ctx }: { ctx: PanelCtx }) {
  const { scene, detections, selectedObject } = ctx;
  const members = useMemo(
    () => detections.filter((d) => selectedObject !== null && d.objectId === selectedObject),
    [detections, selectedObject],
  );
  const [pair, setPair] = useState<{ a: string; b: string } | null>(null);
  const a = detections.find((d) => d.id === (pair?.a ?? members[0]?.id));
  const b = detections.find((d) => d.id === (pair?.b ?? members[1]?.id));
  const exp = a && b && a.id !== b.id ? explainPair(scene.result, a, b) : null;
  const options = detections.map((d) => (
    <option key={d.id} value={d.id}>
      {d.id} · photo {d.view + 1} · {statusLabel(d)}
    </option>
  ));
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Select an object to list its detections, or pick any two detections to see why they were (or were not)
        merged.
      </p>
      <div role="group" aria-label="Objects" className="flex flex-wrap gap-2">
        {scene.result.instances.map((i) => (
          <button
            key={i.id}
            type="button"
            aria-pressed={selectedObject === i.id}
            onClick={() => {
              ctx.onSelectObject(i.id);
              setPair(null);
            }}
            className={cn("rounded-full border border-border px-3 py-1 text-sm", selectedObject === i.id && "bg-accent text-accent-foreground")}
          >
            {objectLabel(i.id)}
          </button>
        ))}
      </div>
      {selectedObject !== null && (
        <section aria-label={`Members of ${objectLabel(selectedObject)}`}>
          <h3 className="text-sm font-semibold">
            {objectLabel(selectedObject)} was seen in {members.length} detections
          </h3>
          <ul className="mt-1 text-sm">
            {members.map((d) => (
              <li key={d.id} className="flex items-center gap-2 py-0.5">
                <span className="tabular-nums">
                  {d.id} (photo {d.view + 1})
                </span>
                <StatusBadge status={d.status} />
              </li>
            ))}
          </ul>
        </section>
      )}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <label className="text-sm">
          Detection A
          <select
            className="mt-1 block w-full rounded border border-border bg-card p-2"
            value={a?.id ?? ""}
            onChange={(e) => setPair({ a: e.target.value, b: b?.id ?? "" })}
          >
            <option value="">Choose…</option>
            {options}
          </select>
        </label>
        <label className="text-sm">
          Detection B
          <select
            className="mt-1 block w-full rounded border border-border bg-card p-2"
            value={b?.id ?? ""}
            onChange={(e) => setPair({ a: a?.id ?? "", b: e.target.value })}
          >
            <option value="">Choose…</option>
            {options}
          </select>
        </label>
      </div>
      {exp ? (
        <section aria-label="Match explanation" data-testid="explanation" className="rounded-md border border-border p-3">
          <p className="font-medium">{exp.sentence}</p>
          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm tabular-nums">
            <dt>3D distance</dt>
            <dd>{exp.distanceInSizes === null ? "n/a" : `${fmt(exp.distanceInSizes)} object sizes`}</dd>
            <dt>Reprojection error</dt>
            <dd>{exp.reprojectionPx === null ? "n/a" : `${fmt(exp.reprojectionPx)} px`}</dd>
            <dt>Appearance similarity</dt>
            <dd>{fmt(exp.appearance)}</dd>
            <dt>Match probability</dt>
            <dd>{fmt(exp.matchProb)}</dd>
            <dt>Decision</dt>
            <dd>{exp.decision === "same" ? "Same object" : "Different objects"}</dd>
            <dt>Rule</dt>
            <dd>{exp.rule}</dd>
          </dl>
        </section>
      ) : (
        <p className="text-sm text-muted-foreground">Choose two different detections.</p>
      )}
    </div>
  );
}

export function WarningsTab({ scene }: { scene: DemoScene }) {
  const { warnings } = scene.result;
  if (!warnings.length) return <p className="text-sm text-muted-foreground">No warnings for this scene.</p>;
  return (
    <ul className="space-y-3">
      {warnings.map((w) => (
        <li key={w.code} className="rounded-md border border-border p-3" data-testid="warning">
          <p className="font-medium">{w.message}</p>
          {w.fix && (
            <p className="mt-1 text-sm text-muted-foreground">
              <span className="font-semibold">How to fix: </span>
              {w.fix}
            </p>
          )}
        </li>
      ))}
    </ul>
  );
}

export function DetailsTab({ scene }: { scene: DemoScene }) {
  const { result, manifest, provenance } = scene;
  const rows: [string, string][] = [
    ["Detector", result.model_versions.detector],
    ["Backbone", result.model_versions.backbone],
    ["Associator", result.model_versions.associator],
    ["Interval method", result.interval.method],
    ["Schema version", result.schema_version],
    ["Recon cache hit", result.recon_cache_hit ? "yes" : "no"],
    ["Peak VRAM", `${result.peak_vram_mb.toFixed(0)} MB`],
    ["Commit", provenance.commit.slice(0, 10)],
    ["Run at (UTC)", provenance.timestamp],
    ["Photos", String(manifest.frames.length)],
    ["Points shown / total", `${manifest.n_points}`],
  ];
  return (
    <div className="space-y-4">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="break-all tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      <section aria-label="Timings per stage">
        <h3 className="text-sm font-semibold">Timings per stage</h3>
        <ul className="mt-1 text-sm tabular-nums">
          {Object.entries(result.timings).map(([k, v]) => (
            <li key={k}>
              {k}: {v.toFixed(3)} s
            </li>
          ))}
        </ul>
      </section>
      <a className="text-sm font-medium text-accent underline" href={`${scene.base}/result.json`}>
        Raw result.json
      </a>
    </div>
  );
}
