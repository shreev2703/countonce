/**
 * Demo asset manifest (`web/public/demo/<scene>/manifest.json`), written by
 * `scripts/build_demo.py` (countonce/export/demo.py). Versions must match the backend:
 * `CURRENT_VERSION` from result.ts, the `CPC1` point format below.
 */
import { z } from "zod";
import { CURRENT_VERSION } from "./result";

export const DEMO_FORMAT_VERSION = "1";
export const POINTS_FORMAT_VERSION = 1;
export const POINTS_HEADER_BYTES = 12;

export const DemoManifest = z.strictObject({
  demo_format_version: z.literal(DEMO_FORMAT_VERSION),
  schema_version: z.literal(CURRENT_VERSION),
  points_format_version: z.literal(POINTS_FORMAT_VERSION),
  scene_id: z.string(),
  label: z.string(),
  prompt: z.string(),
  count: z.number().int().min(0),
  gt_count: z.number().int().nullable(),
  n_points: z.number().int().min(0),
  frames: z.array(z.string()),
  files: z.record(z.string(), z.string().regex(/^[0-9a-f]{64}$/)),
});

export type DemoManifestDoc = z.infer<typeof DemoManifest>;

export const EvidenceFile = z.strictObject({
  schema_version: z.literal(CURRENT_VERSION),
  scene_id: z.string(),
  evidence: z.array(
    z.strictObject({
      det_a: z.string(),
      det_b: z.string(),
      dist_3d_norm: z.number(),
      reproj_error_px: z.number(),
      appearance_sim: z.number(),
      match_prob: z.number(),
      merged: z.boolean(),
      reason: z.string(),
    }),
  ),
});

/** Decode `points.bin` (CPC1): header, float32 xyz (n,3), uint8 rgb (n,3), little endian. */
export function parsePoints(buf: ArrayBuffer): { xyz: Float32Array; rgb: Uint8Array } {
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  const version = view.getUint32(4, true);
  const n = view.getUint32(8, true);
  if (magic !== "CPC1" || version !== POINTS_FORMAT_VERSION) {
    throw new Error(`unsupported points file ${magic} v${version}`);
  }
  if (buf.byteLength !== POINTS_HEADER_BYTES + n * 15) throw new Error("points file has the wrong length");
  const xyz = new Float32Array(buf, POINTS_HEADER_BYTES, n * 3);
  const rgb = new Uint8Array(buf, POINTS_HEADER_BYTES + n * 12, n * 3);
  return { xyz, rgb };
}
