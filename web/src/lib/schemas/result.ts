/**
 * Zod mirror of `countonce/schemas/__init__.py` (docs/PLAN.md 8.10, 17.9).
 *
 * `result.json` / `markers.json` carry `schema_version`. `parseResult` / `parseMarkers` pick the
 * schema for that version, upgrade older documents the same way the backend does, and always
 * return the CURRENT shape, so old shared links keep rendering. Schemas are strict like the
 * Pydantic models (unknown keys are an error: frontend and backend must not drift silently).
 * Keep `CURRENT_VERSION` equal to the backend; `schemas.test.ts` checks the stored fixtures.
 */
import { z } from "zod";

export const CURRENT_VERSION = "1.1";

const xyz = z.tuple([z.number(), z.number(), z.number()]);

export const Interval = z.strictObject({
  low: z.number().int(),
  high: z.number().int(),
  level: z.number().gt(0).lt(1),
  method: z.string(),
  n_calibration: z.number().int().nullable().optional(),
});

export const ModelVersions = z.strictObject({
  detector: z.string(),
  backbone: z.string(),
  associator: z.string(),
  calibration: z.string().nullable().optional(),
});

export const Instance = z.strictObject({
  id: z.number().int(),
  xyz,
  size: z.number(),
  members: z.array(z.string()),
  n_views: z.number().int(),
  conf: z.number().min(0).max(1),
  possibly_moving: z.boolean().default(false),
});

export const MatchEvidence = z.strictObject({
  det_a: z.string(),
  det_b: z.string(),
  dist_3d_norm: z.number(),
  reproj_error_px: z.number(),
  appearance_sim: z.number(),
  match_prob: z.number(),
  merged: z.boolean(),
  reason: z.string(),
});

export const DetectionRecord = z.strictObject({
  id: z.string(),
  frame_id: z.number().int(),
  view: z.number().int(),
  xyz: xyz.nullable().optional(),
  size: z.number().nullable().optional(),
  uv: z.tuple([z.number(), z.number()]).nullable().optional(),
  feature: z.array(z.number()).nullable().optional(),
});

export const CameraRecord = z.strictObject({
  view: z.number().int(),
  intrinsics: z.array(z.array(z.number())),
  extrinsics: z.array(z.array(z.number())),
});

export const Warning = z.strictObject({
  code: z.string(),
  message: z.string(),
  fix: z.string().nullable().optional(),
});

const resultV1Shape = {
  prompt: z.string(),
  negative_prompts: z.array(z.string()).default([]),
  count: z.number().int().min(0),
  zero_count: z.boolean().default(false),
  interval: Interval,
  confidence_method: z.string(),
  model_versions: ModelVersions,
  input_hash: z.string(),
  recon_cache_key: z.string().nullable().optional(),
  recon_cache_hit: z.boolean().default(false),
  oom_fallback_rung: z.number().int().default(0),
  peak_vram_mb: z.number().default(0),
  timings: z.record(z.string(), z.number()).default({}),
  config: z.record(z.string(), z.unknown()).default({}),
  instances: z.array(Instance).default([]),
  per_frame_counts: z.record(z.string(), z.number().int()).default({}),
  detection_status: z.record(z.string(), z.string()).default({}),
  reject_reasons: z.record(z.string(), z.string()).default({}),
  match_evidence: z.array(MatchEvidence).default([]),
  warnings: z.array(Warning).default([]),
};

export const ResultV1 = z.strictObject({ schema_version: z.literal("1.0"), ...resultV1Shape });

export const ResultV11 = z.strictObject({
  ...resultV1Shape,
  schema_version: z.literal("1.1"),
  detections: z.array(DetectionRecord).default([]),
  cameras: z.array(CameraRecord).default([]),
});

export const Marker = z.strictObject({
  id: z.number().int(),
  xyz,
  size: z.number(),
  conf: z.number().min(0).max(1),
  n_views: z.number().int(),
  members: z.array(z.string()),
  possibly_moving: z.boolean().default(false),
});

// Backend `MarkersV1` is one model used for both versions (only schema_version differs).
export const Markers = z.strictObject({
  schema_version: z.enum(["1.0", "1.1"]),
  count: z.number().int().min(0),
  interval_low: z.number().int(),
  interval_high: z.number().int(),
  scene_scale: z.number().nullable().optional(),
  markers: z.array(Marker).default([]),
});

export type Result = z.infer<typeof ResultV11>;
export type MarkersDoc = z.infer<typeof Markers>;

export class SchemaVersionError extends Error {
  constructor(public readonly version: unknown) {
    super(`unsupported schema_version ${JSON.stringify(version)}`);
    this.name = "SchemaVersionError";
  }
}

export const SUPPORTED_VERSIONS = ["1.0", "1.1"] as const;

function versionOf(raw: unknown): string {
  const v = typeof raw === "object" && raw !== null ? (raw as Record<string, unknown>).schema_version : undefined;
  if (typeof v !== "string" || !(SUPPORTED_VERSIONS as readonly string[]).includes(v)) {
    throw new SchemaVersionError(v);
  }
  return v;
}

/** Version -> upgrade of a raw dict one step. Mirror of `UPGRADERS` in the backend. */
const UPGRADERS: Record<string, (raw: Record<string, unknown>) => Record<string, unknown>> = {
  "1.0": (raw) => {
    const out: Record<string, unknown> = { ...raw, schema_version: "1.1" };
    if ("instances" in out) {
      out.detections ??= [];
      out.cameras ??= [];
    }
    return out;
  },
};

function upgrade(raw: unknown): Record<string, unknown> {
  let doc = { ...(raw as Record<string, unknown>) };
  let version = versionOf(doc);
  while (version !== CURRENT_VERSION) {
    const step = UPGRADERS[version];
    if (!step) throw new SchemaVersionError(version);
    doc = step(doc);
    version = versionOf(doc);
  }
  return doc;
}

/** Validate (and upgrade, if old) a raw `result.json`. Throws SchemaVersionError / ZodError. */
export function parseResult(raw: unknown): Result {
  versionOf(raw);
  return ResultV11.parse(upgrade(raw));
}

/** Validate (and upgrade, if old) a raw `markers.json`. */
export function parseMarkers(raw: unknown): MarkersDoc {
  versionOf(raw);
  return Markers.parse(upgrade(raw));
}
