/** Demo scene types and the lazy point-cloud loader (client-safe: no zod). Scene JSON is loaded by `demo-server.ts`. */
import { parsePoints } from "@/lib/schemas/points";
import type { DemoManifest } from "@/lib/schemas/demo";
import type { MarkersDoc, Result } from "@/lib/schemas/result";

export interface DemoScene {
  base: string;
  manifest: ReturnType<typeof DemoManifest.parse>;
  result: Result;
  markers: MarkersDoc;
  provenance: { synthetic: boolean; commit: string; timestamp: string };
}

/**
 * Fetch and decode the point cloud. Kept apart from the scene so the 3D payload is only requested
 * once the viewer is activated (first paint and the count need no points). `parsePoints` only wraps
 * typed-array views over the buffer, so there is no heavy decode to move to a worker.
 */
export async function loadDemoPoints(base: string): Promise<{ xyz: Float32Array; rgb: Uint8Array }> {
  const r = await fetch(`${base}/points.bin`);
  if (!r.ok) throw new Error(`points.bin: HTTP ${r.status}`);
  return parsePoints(await r.arrayBuffer());
}
