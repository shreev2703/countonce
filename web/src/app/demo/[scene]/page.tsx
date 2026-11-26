import { readdirSync } from "node:fs";
import { join } from "node:path";

import { ResultsExperience } from "@/components/result-view/results-experience";
import { loadDemoSceneServer } from "@/lib/result-view/demo-server";

export const dynamicParams = false;

/** One static page per exported demo scene under `public/demo/` (no backend needed). */
export function generateStaticParams() {
  const dir = join(process.cwd(), "public", "demo");
  return readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => ({ scene: e.name }));
}

export default async function DemoPage({ params }: { params: Promise<{ scene: string }> }) {
  const { scene } = await params;
  return <ResultsExperience scene={loadDemoSceneServer(scene)} />;
}
