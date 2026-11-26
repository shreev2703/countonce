import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { parseResult } from "@/lib/schemas/result";
import {
  buildCameras,
  buildDetections,
  cameraFovDeg,
  confidenceBadge,
  csvForResult,
  explainPair,
  frustumCorners,
  intervalText,
  lodStride,
  stepObject,
} from "./model";

const raw = JSON.parse(readFileSync(join(process.cwd(), "public/demo/tiny/result.json"), "utf8"));
const result = parseResult(raw);

describe("results model on the tiny demo", () => {
  const dets = buildDetections(result);
  const cams = buildCameras(result);

  it("assigns every detection to an object and 5 are NEW", () => {
    expect(dets).toHaveLength(25);
    expect(dets.every((d) => d.objectId !== null)).toBe(true);
    expect(dets.filter((d) => d.status === "new")).toHaveLength(result.count);
  });

  it("recovers camera centres at the same distance from the origin", () => {
    const r = cams.map((c) => Math.hypot(...c.center));
    for (const x of r) expect(x).toBeCloseTo(4.472136, 4);
    // cameras look at the origin
    for (const c of cams) {
      const toOrigin = c.center.map((v) => -v / Math.hypot(...c.center));
      const dot = toOrigin.reduce((s, v, k) => s + v * (c.forward[k] ?? 0), 0);
      expect(dot).toBeCloseTo(1, 4);
    }
  });

  it("frustum corners lie in front of the camera", () => {
    const c = cams[0]!;
    for (const p of frustumCorners(c, 1)) {
      const d = p.map((v, k) => v - (c.center[k] ?? 0));
      expect(d.reduce((s, v, k) => s + v * (c.forward[k] ?? 0), 0)).toBeCloseTo(1, 5);
    }
    expect(cameraFovDeg(c)).toBeGreaterThan(30);
  });

  it("explains a merged pair and a same-photo pair", () => {
    const same = explainPair(result, dets.find((d) => d.id === "0:0")!, dets.find((d) => d.id === "1:1")!);
    expect(same.decision).toBe("same");
    expect(same.matchProb).not.toBeNull();
    const diff = explainPair(result, dets.find((d) => d.id === "0:0")!, dets.find((d) => d.id === "0:1")!);
    expect(diff.decision).toBe("different");
    expect(diff.rule).toMatch(/Same photo/);
  });

  it("formats interval, confidence and CSV", () => {
    expect(intervalText(result)).toContain("likely 5–25, 80% range");
    expect(confidenceBadge(result).level).toBe("Low");
    expect(csvForResult(result).trim().split("\n")).toHaveLength(result.count + 1);
  });
});

describe("helpers", () => {
  it("steps through objects with wrap-around", () => {
    expect(stepObject([0, 1, 2], null, 1)).toBe(0);
    expect(stepObject([0, 1, 2], null, -1)).toBe(2);
    expect(stepObject([0, 1, 2], 2, 1)).toBe(0);
    expect(stepObject([0, 1, 2], 0, -1)).toBe(2);
    expect(stepObject([], null, 1)).toBeNull();
  });
  it("lod stride caps the number of drawn points", () => {
    expect(lodStride(100, 300)).toBe(1);
    const n = 1_200_000;
    expect(Math.ceil(n / lodStride(n, 300_000))).toBeLessThanOrEqual(300_000);
  });
});
