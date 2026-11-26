/** Server-side loader for one precomputed demo scene (static build, no backend, PLAN 17.4). */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { DemoManifest, EvidenceFile } from "@/lib/schemas/demo";
import { parseMarkers, parseResult } from "@/lib/schemas/result";

import type { DemoScene } from "./demo";

export function loadDemoSceneServer(scene: string): DemoScene {
  const dir = join(process.cwd(), "public", "demo", scene);
  const json = (name: string): unknown => JSON.parse(readFileSync(join(dir, name), "utf8"));
  const manifest = DemoManifest.parse(json("manifest.json"));
  EvidenceFile.parse(json("evidence.json"));
  const prov = json("provenance.json") as { synthetic?: boolean; run?: { git_commit?: string; timestamp_utc?: string } };
  return {
    base: `/demo/${encodeURIComponent(scene)}`,
    manifest,
    result: parseResult(json("result.json")),
    markers: parseMarkers(json("markers.json")),
    provenance: {
      synthetic: Boolean(prov.synthetic),
      commit: prov.run?.git_commit ?? "unknown",
      timestamp: prov.run?.timestamp_utc ?? "unknown",
    },
  };
}
