"use client";

import { useQuery } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type * as THREE from "three";

import { loadDemoScene } from "@/lib/result-view/demo";
import {
  buildCameras,
  buildDetections,
  MAX_POINTS_DESKTOP,
  MAX_POINTS_PHONE,
  objectLabel,
  stepObject,
} from "@/lib/result-view/model";
import { cn } from "@/lib/utils";

import { CountCard } from "./count-card";
import { PhotoTile } from "./photo-tile";
import { DetailsTab, ExplainTab, PhotosTab, WarningsTab, type PanelCtx } from "./panels";
import type { ViewCommand } from "./viewer";
import { ViewerBoundary } from "./viewer-boundary";

const Viewer = dynamic(() => import("./viewer"), {
  ssr: false,
  loading: () => (
    <div role="progressbar" aria-label="Loading 3D scene" className="grid h-full place-items-center text-sm text-muted-foreground">
      Loading 3D scene…
    </div>
  ),
});

const TABS = ["Photos", "Explain", "Warnings", "Details"] as const;
type Tab = (typeof TABS)[number];

/** Media query as external store: false on the server, live value on the client. */
function useMedia(query: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(query);
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

export function ResultsExperience({ scene: sceneId }: { scene: string }) {
  const q = useQuery({ queryKey: ["demo", sceneId], queryFn: () => loadDemoScene(sceneId), staleTime: Infinity });
  const scene = q.data;

  const [tab, setTab] = useState<Tab>("Photos");
  const [selectedObject, setSelectedObject] = useState<number | null>(null);
  const [hoverObject, setHoverObject] = useState<number | null>(null);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  const [showRejected, setShowRejected] = useState(false);
  const [command, setCommand] = useState<ViewCommand>({ kind: "reset", n: 0 });
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const phone = useMedia("(pointer: coarse), (max-width: 767px)");
  const maxPoints = phone ? MAX_POINTS_PHONE : MAX_POINTS_DESKTOP;
  const reducedMotion = useMedia("(prefers-reduced-motion: reduce)");

  const cameras = useMemo(() => (scene ? buildCameras(scene.result) : []), [scene]);
  const detections = useMemo(() => (scene ? buildDetections(scene.result) : []), [scene]);
  const rejected = useMemo(() => detections.filter((d) => d.status === "rejected"), [detections]);
  const ids = useMemo(() => scene?.markers.markers.map((m) => m.id) ?? [], [scene]);
  const highlightObject = hoverObject ?? selectedObject;

  const run = useCallback((kind: ViewCommand["kind"], id?: number) => setCommand((c) => ({ kind, id, n: c.n + 1 })), []);
  const selectObject = useCallback(
    (id: number | null) => {
      setSelectedObject(id);
      if (id !== null) run("object", id);
    },
    [run],
  );
  const selectCamera = useCallback(
    (view: number | null) => {
      setSelectedCamera(view);
      if (view !== null) run("camera", view);
      else run("reset");
    },
    [run],
  );

  const onKeyDown = (e: React.KeyboardEvent) => {
    if ((e.target as HTMLElement).closest("select, input")) return;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      selectObject(stepObject(ids, selectedObject, 1));
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      selectObject(stepObject(ids, selectedObject, -1));
    } else if (e.key === "Escape") {
      setSelectedObject(null);
      if (selectedCamera !== null) selectCamera(null);
    } else if (e.key.toLowerCase() === "t") run("top");
    else if (e.key.toLowerCase() === "r") {
      setSelectedCamera(null);
      run("reset");
    }
  };

  const screenshot = () => {
    const gl = glRef.current;
    if (!gl) return;
    const a = document.createElement("a");
    a.href = gl.domElement.toDataURL("image/png");
    a.download = `${sceneId}-view.png`;
    a.click();
  };
  const fullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void frameRef.current?.requestFullscreen?.();
  };
  const onReady = useCallback((gl: THREE.WebGLRenderer) => {
    glRef.current = gl;
  }, []);

  if (q.isError) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-12" role="alert">
        <h1 className="text-2xl font-semibold">Could not load this example</h1>
        <p className="mt-2 text-muted-foreground">{(q.error as Error).message}</p>
      </main>
    );
  }
  if (!scene) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-12">
        <p role="status">Loading example…</p>
      </main>
    );
  }

  const ctx: PanelCtx = {
    scene,
    cameras,
    detections,
    selectedObject,
    highlightObject,
    onSelectObject: selectObject,
    onHoverObject: setHoverObject,
  };
  const camPhoto = selectedCamera !== null ? scene.manifest.frames[selectedCamera] : null;
  const toolBtn = "rounded border border-border bg-card px-2.5 py-1 text-sm";

  return (
    <main className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6">
      <header>
        <h1 className="text-2xl font-semibold">{scene.manifest.label}</h1>
        <p className="text-sm text-muted-foreground" data-testid="honesty">
          Precomputed example. Results produced by CountOnce (schema {scene.result.schema_version}, commit{" "}
          {scene.provenance.commit.slice(0, 7)}) on {scene.provenance.timestamp.slice(0, 10)}.
          {scene.provenance.synthetic && " This is a synthetic test scene, not a real photo capture."}
        </p>
      </header>
      <CountCard scene={scene} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[3fr_2fr]">
        <section aria-label="3D viewer" className="flex flex-col gap-2">
          <div
            ref={frameRef}
            tabIndex={0}
            role="application"
            aria-label={`3D view of ${scene.markers.markers.length} counted objects. Arrow keys step through objects, T top view, R reset, Escape clears.`}
            onKeyDown={onKeyDown}
            data-testid="viewer"
            data-selected={selectedObject ?? ""}
            data-camera={selectedCamera ?? ""}
            className="relative h-[55vh] min-h-72 w-full overflow-hidden rounded-lg bg-muted lg:h-[60vh]"
          >
            <ViewerBoundary>
              <Viewer
                xyz={scene.xyz}
                rgb={scene.rgb}
                maxPoints={maxPoints}
                markers={scene.markers.markers}
                cameras={cameras}
                rejected={rejected}
                showRejected={showRejected}
                selectedObject={selectedObject}
                highlightObject={highlightObject}
                selectedCamera={selectedCamera}
                command={command}
                reducedMotion={reducedMotion}
                onSelectObject={selectObject}
                onHoverObject={setHoverObject}
                onSelectCamera={selectCamera}
                onReady={onReady}
              />
            </ViewerBoundary>
            {camPhoto && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                data-testid="camera-photo"
                src={`${scene.base}/thumbs/${camPhoto}.jpg`}
                alt={`What photo ${(selectedCamera ?? 0) + 1} saw`}
                className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-60 transition-opacity duration-500 motion-reduce:transition-none [image-rendering:pixelated]"
              />
            )}
          </div>
          <div className="flex flex-wrap gap-2" role="toolbar" aria-label="Viewer controls">
            <button type="button" className={toolBtn} onClick={() => { setSelectedCamera(null); run("reset"); }}>Reset</button>
            <button type="button" className={toolBtn} onClick={() => run("top")}>Top view</button>
            <button type="button" className={toolBtn} onClick={screenshot}>Screenshot</button>
            <button type="button" className={toolBtn} onClick={fullscreen}>Fullscreen</button>
            <label className="flex items-center gap-1.5 text-sm">
              <input
                type="checkbox"
                checked={showRejected}
                disabled={rejected.length === 0}
                onChange={(e) => setShowRejected(e.target.checked)}
              />
              Rejected detections ({rejected.length})
            </label>
          </div>
          <div role="group" aria-label="Counted objects" className="flex flex-wrap gap-2">
            {scene.markers.markers.map((m) => (
              <button
                key={m.id}
                type="button"
                data-testid={`object-${m.id}`}
                aria-pressed={selectedObject === m.id}
                onClick={() => selectObject(selectedObject === m.id ? null : m.id)}
                onPointerEnter={() => setHoverObject(m.id)}
                onPointerLeave={() => setHoverObject(null)}
                className={cn(
                  "rounded-full border border-border px-3 py-1 text-sm tabular-nums",
                  highlightObject === m.id && "bg-accent text-accent-foreground",
                )}
              >
                Object {objectLabel(m.id)}
              </button>
            ))}
          </div>
        </section>

        <section aria-label="Details panel">
          <div role="tablist" aria-label="Result details" className="mb-3 flex gap-1 border-b border-border">
            {TABS.map((t, i) => (
              <button
                key={t}
                role="tab"
                id={`tab-${t}`}
                type="button"
                aria-selected={tab === t}
                aria-controls={`panel-${t}`}
                tabIndex={tab === t ? 0 : -1}
                onClick={() => setTab(t)}
                onKeyDown={(e) => {
                  const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
                  if (!d) return;
                  e.stopPropagation();
                  const next = TABS[(i + d + TABS.length) % TABS.length] ?? "Photos";
                  setTab(next);
                  document.getElementById(`tab-${next}`)?.focus();
                }}
                className={cn(
                  "-mb-px border-b-2 px-3 py-2 text-sm font-medium",
                  tab === t ? "border-accent text-foreground" : "border-transparent text-muted-foreground",
                )}
              >
                {t}
                {t === "Warnings" && scene.result.warnings.length > 0 && ` (${scene.result.warnings.length})`}
              </button>
            ))}
          </div>
          <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="max-h-[60vh] overflow-y-auto pr-1">
            {tab === "Photos" && <PhotosTab ctx={ctx} />}
            {tab === "Explain" && <ExplainTab ctx={ctx} />}
            {tab === "Warnings" && <WarningsTab scene={scene} />}
            {tab === "Details" && <DetailsTab scene={scene} />}
          </div>
        </section>
      </div>

      <section aria-label="Photo strip">
        <ul className="flex gap-2 overflow-x-auto pb-1">
          {cameras.map((cam) => (
            <li key={cam.view} className="w-40 shrink-0">
              <button
                type="button"
                data-testid={`strip-${cam.view}`}
                aria-pressed={selectedCamera === cam.view}
                aria-label={`Photo ${cam.view + 1}: fly to this camera`}
                onClick={() => selectCamera(selectedCamera === cam.view ? null : cam.view)}
                className={cn("block w-full rounded-md ring-offset-2", selectedCamera === cam.view && "ring-2 ring-accent")}
              >
                <PhotoTile
                  base={scene.base}
                  frame={scene.manifest.frames[cam.view] ?? ""}
                  cam={cam}
                  detections={detections.filter((d) => d.view === cam.view)}
                  highlightObject={highlightObject}
                  onSelectObject={() => {}}
                  onHoverObject={() => {}}
                  interactive={false}
                  className="pointer-events-none"
                />
              </button>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
