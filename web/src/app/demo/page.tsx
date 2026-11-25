import type { Metadata } from "next";

import { DemoSceneCard } from "@/components/chrome/demo-card";
import { PageShell } from "@/components/chrome/site-chrome";
import { listDemoScenes } from "@/lib/demo-index";

export const metadata: Metadata = {
  title: "Demo gallery · CountOnce",
  description: "Precomputed CountOnce examples you can open without signing in.",
};

export default function DemoGallery() {
  const scenes = listDemoScenes();
  return (
    <PageShell>
      <div className="mx-auto max-w-5xl px-4 py-10">
        <h1 className="text-3xl font-semibold">Demo gallery</h1>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Every example below is a precomputed run of the real pipeline, served as static files. No
          sign-in and no backend needed. Open one to orbit the 3D scene and see why each object was
          counted once.
        </p>
        {scenes.length === 0 ? (
          <p className="mt-8 rounded-lg border border-dashed border-border p-6 text-muted-foreground">
            No demo scenes have been exported yet. Run <code>scripts/build_demo.py</code>.
          </p>
        ) : (
          <ul className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {scenes.map((s) => (
              <li key={s.id}>
                <DemoSceneCard scene={s} />
              </li>
            ))}
          </ul>
        )}
        <p className="mt-8 text-sm text-muted-foreground">
          Real-world scenes (Fuji-SfM trees, own captures) are added once they have been run and
          reviewed; none are shown until then.
        </p>
      </div>
    </PageShell>
  );
}
