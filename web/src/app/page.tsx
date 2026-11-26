import Link from "next/link";

import { DemoSceneCard } from "@/components/chrome/demo-card";
import { PageShell } from "@/components/chrome/site-chrome";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { listDemoScenes } from "@/lib/demo-index";

const PROOF_POINTS = [
  {
    title: "More accurate than adding up per-photo counts",
    body: "Counting each photo and summing double-counts every object seen twice.",
  },
  {
    title: "Works from 3–40 photos, no depth sensor",
    body: "Ordinary phone photos in, one count out.",
  },
  {
    title: "Open benchmark with confidence intervals",
    body: "Methods are compared on public data, with uncertainty reported.",
  },
];

const STEPS = [
  ["Detect", "An open-vocabulary detector finds your target in every photo."],
  ["Lift to 3D", "A feed-forward model places each detection in a shared 3D scene."],
  ["Merge", "Detections that are the same physical object are merged, with the reason recorded."],
  ["Count", "You get one number, an interval, and a confidence level."],
];

export default function Home() {
  const scenes = listDemoScenes();
  const hero = scenes[0];
  return (
    <PageShell>
      <section className="mx-auto grid max-w-5xl gap-8 px-4 py-12 md:grid-cols-2 md:items-center">
        <div>
          <h1 className="text-4xl font-semibold leading-tight">
            Count anything from a few photos, without counting it twice.
          </h1>
          <p className="mt-4 text-lg text-muted-foreground">
            CountOnce merges what every photo saw into one 3D scene, so an object that appears in
            five photos is counted once.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button asChild size="lg">
              <Link href={hero ? `/demo/${hero.id}` : "/demo"}>Try a live example</Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/new">Count your own photos</Link>
            </Button>
          </div>
          <p className="mt-2 text-sm text-muted-foreground">Counting your own photos needs a Google sign-in.</p>
        </div>
        {hero?.thumb ? (
          <figure>
            {/* eslint-disable-next-line @next/next/no-img-element -- tiny static thumbnail */}
            <img
              src={hero.thumb}
              alt={`One of ${hero.nFrames} photos from the ${hero.label}, with detections drawn on it`}
              width={160}
              height={120}
              fetchPriority="high"
              className="aspect-[4/3] w-full rounded-lg border border-border object-cover shadow-md"
            />
            <figcaption className="mt-2 text-sm text-muted-foreground">
              {hero.label}: {hero.count} counted. Open it to explore the 3D scene.
            </figcaption>
          </figure>
        ) : null}
      </section>

      <section aria-labelledby="proof" className="mx-auto max-w-5xl px-4 py-8">
        <h2 id="proof" className="text-2xl font-semibold">
          What we claim, and the evidence
        </h2>
        <ul className="mt-4 grid gap-4 md:grid-cols-3">
          {PROOF_POINTS.map((p) => (
            <li key={p.title}>
              <Card className="h-full p-5">
                <h3 className="font-semibold">{p.title}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{p.body}</p>
                <p className="mt-3 rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
                  No published number yet. This fills in from benchmark results once they are final.
                </p>
              </Card>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="how" className="mx-auto max-w-5xl px-4 py-8">
        <h2 id="how" className="text-2xl font-semibold">
          How it works
        </h2>
        <ol className="mt-4 grid gap-4 md:grid-cols-4">
          {STEPS.map(([title, body], i) => (
            <li key={title}>
              <Card className="h-full p-5">
                <p className="tabular text-sm text-muted-foreground">Step {i + 1}</p>
                <h3 className="font-semibold">{title}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{body}</p>
              </Card>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="gallery" className="mx-auto max-w-5xl px-4 py-8">
        <div className="flex items-baseline justify-between">
          <h2 id="gallery" className="text-2xl font-semibold">
            Demo gallery
          </h2>
          <Link href="/demo" className="text-sm underline underline-offset-4">
            See all
          </Link>
        </div>
        <ul className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {scenes.map((s) => (
            <li key={s.id}>
              <DemoSceneCard scene={s} />
            </li>
          ))}
        </ul>
      </section>
    </PageShell>
  );
}
