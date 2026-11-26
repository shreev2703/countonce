/** Build-time index of the precomputed demo scenes in `public/demo/` (server components only). */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { DemoManifest } from "@/lib/schemas/demo";

export interface DemoCard {
  id: string;
  label: string;
  prompt: string;
  count: number;
  gtCount: number | null;
  nFrames: number;
  thumb: string | null;
  synthetic: boolean;
}

export function listDemoScenes(root = join(process.cwd(), "public", "demo")): DemoCard[] {
  if (!existsSync(root)) return [];
  const cards: DemoCard[] = [];
  for (const e of readdirSync(root, { withFileTypes: true })) {
    if (!e.isDirectory()) continue;
    const dir = join(root, e.name);
    const manifestPath = join(dir, "manifest.json");
    if (!existsSync(manifestPath)) continue;
    const m = DemoManifest.parse(JSON.parse(readFileSync(manifestPath, "utf8")));
    const first = m.frames[0];
    const thumb = first && existsSync(join(dir, "thumbs", `${first}.jpg`))
      ? `/demo/${e.name}/thumbs/${first}.jpg`
      : null;
    let synthetic = false;
    const provPath = join(dir, "provenance.json");
    if (existsSync(provPath)) synthetic = Boolean(JSON.parse(readFileSync(provPath, "utf8")).synthetic);
    cards.push({
      id: m.scene_id,
      label: m.label,
      prompt: m.prompt,
      count: m.count,
      gtCount: m.gt_count ?? null,
      nFrames: m.frames.length,
      thumb,
      synthetic,
    });
  }
  return cards.sort((a, b) => a.id.localeCompare(b.id));
}
