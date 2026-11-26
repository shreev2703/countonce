import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Metadata } from "next";
import Link from "next/link";

import { PageShell } from "@/components/chrome/site-chrome";
import { Walkthrough } from "@/components/explain/walkthrough";
import { listDemoScenes, type DemoCard } from "@/lib/demo-index";
import { parseResult } from "@/lib/schemas/result";

export const metadata: Metadata = {
  title: "How it works · CountOnce",
  description: "Walk through detection, 3D lifting, merging and counting on a real run of the pipeline.",
};

export const dynamic = "force-static";

const fixed = (v: number) => v.toFixed(2);
const photo = (member: string) => Number(member.split(":")[0]) + 1;

function stagesFor(scene: DemoCard) {
  const dir = join(process.cwd(), "public", "demo", scene.id);
  const result = parseResult(JSON.parse(readFileSync(join(dir, "result.json"), "utf8")));
  const manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8")) as { frames: string[] };
  const perPhoto = Object.entries(result.per_frame_counts).sort(([a], [b]) => Number(a) - Number(b));
  const summed = perPhoto.reduce((s, [, n]) => s + n, 0);
  const lifted = result.detections.filter((d) => d.xyz);
  const merged = result.match_evidence.find((e) => e.merged);

  return [
    {
      title: "Detect",
      content: (
        <>
          <p className="mt-2 text-muted-foreground">
            A detector looks for “{result.prompt}” in each photo on its own. It found {summed} boxes across{" "}
            {perPhoto.length} photos.
          </p>
          <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
            {manifest.frames.map((frame, n) => (
              <li key={frame}>
                <figure>
                  {/* eslint-disable-next-line @next/next/no-img-element -- tiny static thumbnail */}
                  <img src={`/demo/${scene.id}/thumbs/${frame}.jpg`} alt={`Photo ${n + 1} with detections drawn on it`} width={160} height={120} className="w-full rounded-md border border-border" />
                  <figcaption className="text-sm text-muted-foreground">
                    Photo {n + 1}: {result.per_frame_counts[String(n)] ?? 0} found
                  </figcaption>
                </figure>
              </li>
            ))}
          </ul>
        </>
      ),
    },
    {
      title: "Lift to 3D",
      content: (
        <>
          <p className="mt-2 text-muted-foreground">
            The 3D reconstruction gives every photo a camera position and a depth, so each box becomes a point with a
            size in one shared scene. {lifted.length} of {result.detections.length} detections were placed. The first
            photo’s, in scene units:
          </p>
          <table className="tabular mt-3 w-full max-w-lg text-sm">
            <thead>
              <tr className="border-b border-border text-left">
                <th scope="col" className="py-1 pr-4">Detection</th>
                <th scope="col" className="py-1 pr-4">Position (x, y, z)</th>
                <th scope="col" className="py-1">Size</th>
              </tr>
            </thead>
            <tbody>
              {lifted
                .filter((d) => d.view === 0)
                .map((d, n) => (
                  <tr key={d.id} className="border-b border-border">
                    <td className="py-1 pr-4">Photo 1, box {n + 1}</td>
                    <td className="py-1 pr-4">{d.xyz?.map(fixed).join(", ")}</td>
                    <td className="py-1">{d.size == null ? "unknown" : fixed(d.size)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </>
      ),
    },
    {
      title: "Merge",
      content: (
        <>
          <p className="mt-2 text-muted-foreground">
            Detections that land on the same physical object are merged. Two boxes in one photo can never be the same
            object. Each line below is one object and the photos it was seen in.
          </p>
          <ul className="tabular mt-3 text-sm">
            {result.instances.map((inst, n) => (
              <li key={inst.id} className="border-b border-border py-1">
                Object {n + 1}: photos {inst.members.map(photo).join(", ")}
              </li>
            ))}
          </ul>
          {merged && (
            <p className="mt-3 text-sm text-muted-foreground">
              One recorded decision: a box in photo {photo(merged.det_a)} and a box in photo {photo(merged.det_b)} were
              merged. They were {fixed(merged.dist_3d_norm)} object sizes apart, with match probability{" "}
              {fixed(merged.match_prob)}.
            </p>
          )}
        </>
      ),
    },
    {
      title: "Count",
      content: (
        <>
          <p className="mt-2 text-muted-foreground">
            The count is the number of merged objects. Adding up the photos would have given {summed}.
          </p>
          <p className="tabular mt-3 text-4xl font-semibold">{result.count}</p>
          <p className="text-sm text-muted-foreground">
            {scene.gtCount === null ? "" : `True number: ${scene.gtCount}. `}
            Range from changing the settings: {result.interval.low} to {result.interval.high}. A wide range like this
            is a reason to check the result.
          </p>
          <p className="mt-4">
            <Link href={`/demo/${scene.id}`} className="underline underline-offset-4">
              Explore this scene in 3D
            </Link>
          </p>
        </>
      ),
    },
  ];
}

export default function HowItWorks() {
  const scene = listDemoScenes()[0];
  return (
    <PageShell>
      <div className="mx-auto max-w-5xl px-4 py-10">
        <h1 className="text-3xl font-semibold">How it works</h1>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Four stages turn a handful of photos into one count. Step through them on a recorded run.
        </p>
        {scene ? (
          <>
            {scene.synthetic && (
              <p className="my-4 max-w-2xl rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
                This example is a synthetic test scene, run with a simple colour detector and known geometry. It shows
                what each stage does, not how accurate CountOnce is on real photos. Real photos use GroundingDINO for
                detection and VGGT for reconstruction.
              </p>
            )}
            <Walkthrough stages={stagesFor(scene)} />
          </>
        ) : (
          <p className="mt-6 rounded-lg border border-dashed border-border p-6 text-muted-foreground">
            No example has been exported yet.
          </p>
        )}
      </div>
    </PageShell>
  );
}
