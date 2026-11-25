/** Loads and validates one precomputed demo scene from `/demo/<scene>/` (no backend, PLAN 17.4). */
import { DemoManifest, EvidenceFile, parsePoints } from "@/lib/schemas/demo";
import { parseMarkers, parseResult, type MarkersDoc, type Result } from "@/lib/schemas/result";

export interface DemoScene {
  base: string;
  manifest: ReturnType<typeof DemoManifest.parse>;
  result: Result;
  markers: MarkersDoc;
  xyz: Float32Array;
  rgb: Uint8Array;
  provenance: { synthetic: boolean; commit: string; timestamp: string };
}

async function getJson(url: string): Promise<unknown> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

export async function loadDemoScene(scene: string): Promise<DemoScene> {
  const base = `/demo/${encodeURIComponent(scene)}`;
  const [manifestRaw, resultRaw, markersRaw, evidenceRaw, provRaw, pointsRes] = await Promise.all([
    getJson(`${base}/manifest.json`),
    getJson(`${base}/result.json`),
    getJson(`${base}/markers.json`),
    getJson(`${base}/evidence.json`),
    getJson(`${base}/provenance.json`),
    fetch(`${base}/points.bin`),
  ]);
  if (!pointsRes.ok) throw new Error(`points.bin: HTTP ${pointsRes.status}`);
  const manifest = DemoManifest.parse(manifestRaw);
  EvidenceFile.parse(evidenceRaw);
  const { xyz, rgb } = parsePoints(await pointsRes.arrayBuffer());
  const prov = provRaw as { synthetic?: boolean; run?: { git_commit?: string; timestamp_utc?: string } };
  return {
    base,
    manifest,
    result: parseResult(resultRaw),
    markers: parseMarkers(markersRaw),
    xyz,
    rgb,
    provenance: {
      synthetic: Boolean(prov.synthetic),
      commit: prov.run?.git_commit ?? "unknown",
      timestamp: prov.run?.timestamp_utc ?? "unknown",
    },
  };
}
