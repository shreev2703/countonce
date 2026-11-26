/**
 * Pure view-model helpers for the results experience (PLAN 17.5). No React, no three.js:
 * everything here is unit-tested in model.test.ts.
 */
import type { MarkersDoc, Result } from "@/lib/schemas/result";

export type Status = "new" | "duplicate" | "rejected";
export type Vec3 = [number, number, number];

export interface Detection {
  id: string;
  view: number;
  uv: [number, number] | null;
  xyz: Vec3 | null;
  size: number | null;
  status: Status;
  /** Object number this detection belongs to (null when rejected or unassigned). */
  objectId: number | null;
  reason: string | null;
}

export interface CameraPose {
  view: number;
  center: Vec3;
  /** Unit viewing direction in world coordinates. */
  forward: Vec3;
  up: Vec3;
  /** Image size in pixels, from the principal point (assumes it is centred). */
  width: number;
  height: number;
  fx: number;
  fy: number;
}

const v3 = (a: readonly number[]): Vec3 => [a[0] ?? 0, a[1] ?? 0, a[2] ?? 0];
const num = (a: readonly number[] | undefined, i: number): number => a?.[i] ?? 0;

export function objectLabel(id: number): string {
  return `#${id + 1}`;
}

/** Extrinsics are world-to-camera [R | t] (OpenCV: x right, y down, z forward). */
export function cameraPose(
  view: number,
  intrinsics: number[][],
  extrinsics: number[][],
): CameraPose {
  const R = [0, 1, 2].map((r) => v3(extrinsics[r] ?? []));
  const t = [0, 1, 2].map((r) => num(extrinsics[r], 3));
  // C = -R^T t; world direction of camera axis a = R^T e_a = row a of R.
  const center = v3([0, 1, 2].map((j) => -(num(R[0], j) * num(t, 0) + num(R[1], j) * num(t, 1) + num(R[2], j) * num(t, 2))));
  const forward = v3(R[2] ?? []);
  const up = v3((R[1] ?? []).map((v) => -v));
  return {
    view,
    center,
    forward,
    up,
    width: Math.round(num(intrinsics[0], 2) * 2 + 1),
    height: Math.round(num(intrinsics[1], 2) * 2 + 1),
    fx: num(intrinsics[0], 0),
    fy: num(intrinsics[1], 1),
  };
}

export function buildCameras(result: Result): CameraPose[] {
  return result.cameras.map((c) => cameraPose(c.view, c.intrinsics, c.extrinsics));
}

export function buildDetections(result: Result): Detection[] {
  const owner = new Map<string, number>();
  for (const inst of result.instances) for (const m of inst.members) owner.set(m, inst.id);
  return result.detections.map((d) => {
    const raw = result.detection_status[d.id];
    const status: Status = raw === "new" || raw === "duplicate" || raw === "rejected" ? raw : "rejected";
    return {
      id: d.id,
      view: d.view,
      uv: d.uv ?? null,
      xyz: (d.xyz as Vec3 | null | undefined) ?? null,
      size: d.size ?? null,
      status,
      objectId: owner.get(d.id) ?? null,
      reason: result.reject_reasons[d.id] ?? null,
    };
  });
}

export type ConfidenceLevel = "High" | "Medium" | "Low";

/** Confidence badge with the reason shown next to it. Thresholds are documented in the UI text. */
export function confidenceBadge(result: Result): { level: ConfidenceLevel; reason: string } {
  const { low, high } = result.interval;
  const width = high - low;
  const rel = result.count > 0 ? width / result.count : width;
  const n = result.warnings.length;
  if (rel > 1) return { level: "Low", reason: `the plausible range (${low}–${high}) is much wider than the count` };
  if (rel > 0.2 || n > 0) {
    const why = rel > 0.2 ? `the range is ${low}–${high}` : `${n} warning${n === 1 ? "" : "s"}`;
    return { level: "Medium", reason: why };
  }
  return { level: "High", reason: "a narrow range and no warnings" };
}

export function intervalText(result: Result): string {
  const { low, high, level, method, n_calibration } = result.interval;
  const pct = Math.round(level * 100);
  const how = method === "sensitivity" ? "sensitivity to settings" : method.replaceAll("_", " ");
  const cal = n_calibration ? `, n = ${n_calibration} calibration scenes` : "";
  return `likely ${low}–${high}, ${pct}% range, ${how}${cal}`;
}

export function statusLabel(d: Detection): string {
  if (d.status === "new") return `NEW ${objectLabel(d.objectId ?? 0)}`;
  if (d.status === "duplicate") return `DUPLICATE of ${objectLabel(d.objectId ?? 0)}`;
  return d.reason ? `REJECTED (${d.reason})` : "REJECTED";
}

export function csvForResult(result: Result): string {
  const rows = ["object,x,y,z,size,n_views,confidence"];
  for (const i of result.instances) {
    rows.push([i.id + 1, ...i.xyz, i.size, i.n_views, i.conf].join(","));
  }
  return rows.join("\n") + "\n";
}

export interface Explanation {
  sentence: string;
  distanceInSizes: number | null;
  reprojectionPx: number | null;
  appearance: number | null;
  matchProb: number | null;
  decision: "same" | "different";
  rule: string;
}

function dist(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/**
 * "Explain this match" for any two detections. Uses the stored pairwise evidence when it exists
 * and otherwise falls back to what can be computed from the detections themselves.
 */
export function explainPair(result: Result, a: Detection, b: Detection): Explanation {
  const ev = result.match_evidence.find(
    (e) => (e.det_a === a.id && e.det_b === b.id) || (e.det_a === b.id && e.det_b === a.id),
  );
  const sameObject = a.objectId !== null && a.objectId === b.objectId;
  let distanceInSizes = ev?.dist_3d_norm ?? null;
  if (distanceInSizes === null && a.xyz && b.xyz) {
    const sizes = [a.size, b.size].filter((s): s is number => typeof s === "number" && s > 0);
    if (sizes.length) distanceInSizes = dist(a.xyz, b.xyz) / (sizes.reduce((p, q) => p + q, 0) / sizes.length);
  }
  let rule: string;
  if (a.view === b.view) rule = "Same photo, can't be the same object (cannot-link).";
  else if (ev) rule = ev.merged ? `Merged by the associator (${ev.reason}).` : `Kept apart (${ev.reason}).`;
  else if (sameObject) rule = "Merged transitively through another detection of the same object.";
  else rule = "No pairwise decision was recorded; they ended up in different objects.";
  const decision = sameObject ? "same" : "different";
  const where = a.view === b.view ? "the same photo" : `photos ${a.view + 1} and ${b.view + 1}`;
  const sentence = sameObject
    ? `The two sightings (${where}) are the same physical object, counted once as ${objectLabel(a.objectId as number)}.`
    : `The two sightings (${where}) are different objects${a.view === b.view ? " because one photo cannot show the same object twice" : ""}.`;
  return {
    sentence,
    distanceInSizes,
    reprojectionPx: ev?.reproj_error_px ?? null,
    appearance: ev?.appearance_sim ?? null,
    matchProb: ev?.match_prob ?? null,
    decision,
    rule,
  };
}

/** Next object id when stepping with the arrow keys (wraps; null selection starts at the ends). */
export function stepObject(ids: number[], current: number | null, dir: 1 | -1): number | null {
  if (!ids.length) return null;
  if (current === null) return (dir === 1 ? ids[0] : ids[ids.length - 1]) ?? null;
  const i = ids.indexOf(current);
  if (i < 0) return ids[0] ?? null;
  return ids[(i + dir + ids.length) % ids.length] ?? null;
}

/** Evenly strided subsample so at most `max` points are drawn (LOD, PLAN 17.5). */
export function lodStride(n: number, max: number): number {
  return n <= max ? 1 : Math.ceil(n / max);
}

export const MAX_POINTS_PHONE = 300_000;
export const MAX_POINTS_DESKTOP = 1_000_000;

export function sceneBounds(points: Float32Array, markers: MarkersDoc["markers"]): { center: Vec3; radius: number } {
  const lo: Vec3 = [Infinity, Infinity, Infinity];
  const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  const add = (v: Vec3) => {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(num(lo, k), num(v, k));
      hi[k] = Math.max(num(hi, k), num(v, k));
    }
  };
  for (let i = 0; i + 2 < points.length; i += 3) add([num(points as unknown as number[], i), num(points as unknown as number[], i + 1), num(points as unknown as number[], i + 2)]);
  for (const m of markers) add(m.xyz);
  if (!Number.isFinite(lo[0])) return { center: [0, 0, 0], radius: 1 };
  const center = v3(lo.map((l, k) => (l + num(hi, k)) / 2));
  const radius = Math.max(0.5, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
  return { center, radius };
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Four image-plane corners of a camera frustum drawn `depth` in front of the camera centre. */
export function frustumCorners(cam: CameraPose, depth: number): Vec3[] {
  const down: Vec3 = [-cam.up[0], -cam.up[1], -cam.up[2]];
  const right = cross(down, cam.forward); // OpenCV: x = y × z
  const hw = (cam.width / 2 / cam.fx) * depth;
  const hh = (cam.height / 2 / cam.fy) * depth;
  const mid = v3(cam.center.map((c, k) => c + num(cam.forward, k) * depth));
  return ([[-1, -1], [1, -1], [1, 1], [-1, 1]] as const).map(([sx, sy]) =>
    v3(mid.map((m, k) => m + num(right, k) * hw * sx - num(cam.up, k) * hh * sy)),
  );
}

/** Vertical field of view (degrees) that reproduces a camera's framing. */
export function cameraFovDeg(cam: CameraPose): number {
  return (2 * Math.atan(cam.height / 2 / cam.fy) * 180) / Math.PI;
}
