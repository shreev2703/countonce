"use client";

import { useQuery } from "@tanstack/react-query";
import { m } from "framer-motion";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import type * as THREE from "three";

import { loadDemoPoints, type DemoScene } from "@/lib/result-view/demo";
import {
  buildCameras,
  buildDetections,
  MAX_POINTS_DESKTOP,
  MAX_POINTS_PHONE,
  objectLabel,
  stepObject,
} from "@/lib/result-view/model";
import { useMedia } from "@/lib/use-media";
import { MotionProvider } from "@/components/motion/motion-provider";
import { cn } from "@/lib/utils";

import { CountCard } from "./count-card";
import { PhotoTile } from "./photo-tile";
import { ExplainTab, PhotosTab, TechnicalDetails, WarningsTab, type PanelCtx } from "./panels";
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

const PHONE_QUERY = "(pointer: coarse), (max-width: 767px)";

/**
 * Media queries are read here, in a client-only mount. Reading them in the page component made the
 * server value differ on phones, which forced a second synchronous render during hydration.
 */
function LiveViewer(props: Omit<ComponentProps<typeof Viewer>, "maxPoints" | "reducedMotion">) {
  const phone = useMedia(PHONE_QUERY);
  const reducedMotion = useMedia("(prefers-reduced-motion: reduce)");
  return <Viewer {...props} maxPoints={phone ? MAX_POINTS_PHONE : MAX_POINTS_DESKTOP} reducedMotion={reducedMotion} />;
}

const TABS = ["Photos", "Explain", "Warnings"] as const;
type Tab = (typeof TABS)[number];

export function ResultsExperience({ scene }: { scene: DemoScene }) {
  const sceneId = decodeURIComponent(scene.base.split("/").pop() ?? "");
  // The 3D viewer (three.js, ~0.4 MB of JS) and points.bin load only once activated: on a click, on
  // the first object/camera selection, or automatically after the page is idle on desktop.
  const [view3d, setView3d] = useState(false);
  const pts = useQuery({
    queryKey: ["demo-points", scene.base],
    queryFn: () => loadDemoPoints(scene.base),
    staleTime: Infinity,
    enabled: view3d,
  });

  const [tab, setTab] = useState<Tab>("Photos");
  const [selectedObject, setSelectedObject] = useState<number | null>(null);
  const [hoverObject, setHoverObject] = useState<number | null>(null);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  const [showRejected, setShowRejected] = useState(false);
  const [inspector, setInspector] = useState(true);
  const [command, setCommand] = useState<ViewCommand>({ kind: "reset", n: 0 });
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const frameRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (view3d || window.matchMedia(PHONE_QUERY).matches) return;
    let id = 0;
    const arm = () => {
      id = window.setTimeout(() => setView3d(true), 1200);
    };
    if (document.readyState === "complete") arm();
    else window.addEventListener("load", arm, { once: true });
    return () => {
      window.removeEventListener("load", arm);
      window.clearTimeout(id);
    };
  }, [view3d]);

  const cameras = useMemo(() => buildCameras(scene.result), [scene]);
  const detections = useMemo(() => buildDetections(scene.result), [scene]);
  const rejected = useMemo(() => detections.filter((d) => d.status === "rejected"), [detections]);
  const ids = useMemo(() => scene.markers.markers.map((m) => m.id), [scene]);
  const highlightObject = hoverObject ?? selectedObject;

  const run = useCallback((kind: ViewCommand["kind"], id?: number) => {
    setView3d(true);
    setCommand((c) => ({ kind, id, n: c.n + 1 }));
  }, []);
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

  return (
    <MotionProvider>
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-6">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight first-letter:uppercase">{scene.manifest.label}</h1>
          <p className="text-sm text-muted-foreground" data-testid="honesty">
            Precomputed example.
            {scene.provenance.synthetic && " This is a synthetic test scene, not a real photo capture."}
          </p>
        </header>
        <CountCard scene={scene} />

        <div className={cn("grid grid-cols-1 gap-4", inspector && "lg:grid-cols-[minmax(0,1fr)_380px]")}>
          <section aria-label="3D viewer" className="flex min-w-0 flex-col gap-2">
            <div ref={frameRef} className="surface relative h-[62vh] min-h-96 w-full overflow-hidden rounded-2xl lg:h-[70vh]">
              <div
                tabIndex={0}
                role="application"
                aria-label={`3D view of ${scene.markers.markers.length} counted objects. Arrow keys step through objects, T top view, R reset, Escape clears.`}
                onKeyDown={onKeyDown}
                data-testid="viewer"
                data-selected={selectedObject ?? ""}
                data-camera={selectedCamera ?? ""}
                className="relative h-full w-full"
              >
                {!pts.data ? (
                  <div className="grid h-full place-items-center p-6 text-center" data-testid="viewer-poster">
                    <div className="flex max-w-sm flex-col items-center gap-3">
                      <p className="text-sm text-muted-foreground">
                        {view3d && !pts.isError
                          ? "Loading 3D scene…"
                          : "The 3D view loads on request so the count and photos appear fast."}
                      </p>
                      {pts.isError && <p role="alert" className="text-sm">Could not load the 3D points.</p>}
                      <button
                        type="button"
                        className="glass-btn"
                        disabled={view3d && !pts.isError}
                        onClick={() => (pts.isError ? void pts.refetch() : setView3d(true))}
                      >
                        {pts.isError ? "Retry 3D view" : "Load 3D view"}
                      </button>
                    </div>
                  </div>
                ) : (
                  <ViewerBoundary>
                    <LiveViewer
                      xyz={pts.data.xyz}
                      rgb={pts.data.rgb}
                      markers={scene.markers.markers}
                      cameras={cameras}
                      rejected={rejected}
                      showRejected={showRejected}
                      selectedObject={selectedObject}
                      highlightObject={highlightObject}
                      selectedCamera={selectedCamera}
                      command={command}
                      onSelectObject={selectObject}
                      onHoverObject={setHoverObject}
                      onSelectCamera={selectCamera}
                      onReady={onReady}
                    />
                  </ViewerBoundary>
                )}
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

              <div className="glass absolute left-3 right-3 top-3 flex flex-wrap items-center gap-1 rounded-xl p-1.5 sm:right-auto" role="toolbar" aria-label="Viewer controls">
                <button type="button" className="glass-btn" onClick={() => { setSelectedCamera(null); run("reset"); }}>Reset</button>
                <button type="button" className="glass-btn" onClick={() => run("top")}>Top view</button>
                <button type="button" className="glass-btn" onClick={screenshot} disabled={!pts.data}>Screenshot</button>
                <button type="button" className="glass-btn" onClick={fullscreen}>Fullscreen</button>
                <button type="button" className="glass-btn max-lg:hidden" aria-expanded={inspector} aria-controls="inspector" onClick={() => setInspector((v) => !v)}>
                  {inspector ? "Hide panel" : "Show panel"}
                </button>
                {rejected.length > 0 && (
                  <label className="flex items-center gap-1.5 px-2 text-sm">
                    <input type="checkbox" checked={showRejected} onChange={(e) => setShowRejected(e.target.checked)} />
                    Show discarded ({rejected.length})
                  </label>
                )}
              </div>

              <div role="group" aria-label="Counted objects" className="glass absolute bottom-3 left-1/2 flex w-max max-w-[calc(100%-24px)] -translate-x-1/2 gap-1 overflow-x-auto rounded-full p-1.5">
                {scene.markers.markers.map((marker) => (
                  <button
                    key={marker.id}
                    type="button"
                    data-testid={`object-${marker.id}`}
                    aria-pressed={selectedObject === marker.id}
                    aria-label={`Object ${objectLabel(marker.id)}`}
                    onClick={() => selectObject(selectedObject === marker.id ? null : marker.id)}
                    onPointerEnter={() => setHoverObject(marker.id)}
                    onPointerLeave={() => setHoverObject(null)}
                    className={cn(
                      "shrink-0 rounded-full px-3 py-1 text-sm tabular-nums transition-colors",
                      highlightObject === marker.id ? "bg-accent text-accent-foreground" : "hover:bg-muted",
                    )}
                  >
                    {objectLabel(marker.id)}
                  </button>
                ))}
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Drag to rotate, scroll or pinch to zoom. Pick a number to fly to that object, or a photo below to stand
              where it was taken.
            </p>
          </section>

          <section id="inspector" aria-label="Details panel" className={cn("surface min-w-0 rounded-2xl p-3", !inspector && "lg:hidden")}>
            <div role="tablist" aria-label="Result details" className="mb-3 flex gap-1 rounded-xl bg-muted p-1">
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
                  className={cn("relative flex-1 whitespace-nowrap rounded-lg px-2 py-1.5 text-sm font-medium", tab === t ? "text-foreground" : "text-muted-foreground")}
                >
                  {tab === t && (
                    <m.span aria-hidden layoutId="tab-indicator" className="absolute inset-0 rounded-lg border border-border bg-white/10" transition={{ type: "spring", stiffness: 480, damping: 38 }} />
                  )}
                  <span className="relative">
                    {t}
                    {t === "Warnings" && scene.result.warnings.length > 0 && ` (${scene.result.warnings.length})`}
                  </span>
                </button>
              ))}
            </div>
            <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} tabIndex={0} className="max-h-[60vh] overflow-y-auto p-1">
              {tab === "Photos" && <PhotosTab ctx={ctx} />}
              {tab === "Explain" && <ExplainTab ctx={ctx} />}
              {tab === "Warnings" && <WarningsTab scene={scene} />}
            </div>
          </section>
        </div>

        <section aria-label="Photo strip">
          <ul className="flex gap-2 overflow-x-auto p-1">
            {cameras.map((cam) => (
              <li key={cam.view} className="w-40 shrink-0">
                <button
                  type="button"
                  data-testid={`strip-${cam.view}`}
                  aria-pressed={selectedCamera === cam.view}
                  aria-label={`Photo ${cam.view + 1}: fly to this camera`}
                  onClick={() => selectCamera(selectedCamera === cam.view ? null : cam.view)}
                  className={cn("block w-full rounded-md transition-transform hover:-translate-y-0.5", selectedCamera === cam.view && "ring-2 ring-accent")}
                >
                  <PhotoTile
                    base={scene.base}
                    frame={scene.manifest.frames[cam.view] ?? ""}
                    cam={cam}
                    detections={detections.filter((d) => d.view === cam.view)}
                    highlightObject={highlightObject}
                    onSelectObject={() => {}}
                    onHoverObject={setHoverObject}
                    interactive={false}
                  />
                </button>
              </li>
            ))}
          </ul>
        </section>

        <details className="surface rounded-2xl p-4" data-testid="tech-details">
          <summary className="cursor-pointer text-sm font-semibold">Technical details</summary>
          <div className="mt-4">
            <TechnicalDetails scene={scene} />
          </div>
        </details>
      </div>
    </MotionProvider>
  );
}
