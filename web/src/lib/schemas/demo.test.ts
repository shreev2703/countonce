import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { DemoManifest, EvidenceFile, parsePoints } from "./demo";
import { parseMarkers, parseResult } from "./result";

const ROOT = path.resolve(__dirname, "../../../public/demo");
const scenes = existsSync(ROOT)
  ? readdirSync(ROOT, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name)
  : [];
const json = (scene: string, f: string) => JSON.parse(readFileSync(path.join(ROOT, scene, f), "utf8"));

describe("demo assets (built by scripts/build_demo.py)", () => {
  it("has at least one scene", () => {
    expect(scenes.length).toBeGreaterThan(0);
  });

  it.each(scenes)("%s validates against the Zod schemas", (scene) => {
    const manifest = DemoManifest.parse(json(scene, "manifest.json"));
    const result = parseResult(json(scene, "result.json"));
    const markers = parseMarkers(json(scene, "markers.json"));
    EvidenceFile.parse(json(scene, "evidence.json"));
    expect(result.schema_version).toBe(manifest.schema_version);
    expect(markers.markers).toHaveLength(result.count);
    expect(manifest.count).toBe(result.count);
    for (const [rel, digest] of Object.entries(manifest.files)) {
      const sum = createHash("sha256").update(readFileSync(path.join(ROOT, scene, rel))).digest("hex");
      expect(sum, rel).toBe(digest);
    }
    const buf = readFileSync(path.join(ROOT, scene, "points.bin"));
    const pts = parsePoints(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(pts.xyz.length).toBe(manifest.n_points * 3);
  });

  it("labels the tiny scene as synthetic", () => {
    if (scenes.includes("tiny")) expect(json("tiny", "manifest.json").label).toBe("synthetic test scene");
  });
});
