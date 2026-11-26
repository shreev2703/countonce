/**
 * Demo asset manifest (`web/public/demo/<scene>/manifest.json`), written by
 * `scripts/build_demo.py` (countonce/export/demo.py). Versions must match the backend:
 * `CURRENT_VERSION` from result.ts, the `CPC1` point format below.
 */
import { z } from "zod";
import { CURRENT_VERSION } from "./result";
import { POINTS_FORMAT_VERSION } from "./points";

export { parsePoints, POINTS_FORMAT_VERSION, POINTS_HEADER_BYTES } from "./points";

export const DEMO_FORMAT_VERSION = "1";

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
