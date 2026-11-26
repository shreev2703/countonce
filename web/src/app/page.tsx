import Link from "next/link";

import { DemoSceneCard } from "@/components/chrome/demo-card";
import { PageShell } from "@/components/chrome/site-chrome";
import { HeroDiagram } from "@/components/landing/hero-diagram";
import { MotionProvider } from "@/components/motion/motion-provider";
import { SpotlightCard } from "@/components/motion/spotlight-card";
import { Button } from "@/components/ui/button";
import { loadFuji, type FujiSummary } from "@/lib/benchmark";
import { listDemoScenes } from "@/lib/demo-index";

const PHOTOS = 10;

/** Evidence lines come from the benchmark summary; a missing value gives the empty state. */
function proofPoints(fuji: FujiSummary | null) {
  const mae = (method: string) =>
    fuji?.cells.find((c) => c.method === method && c.k === PHOTOS && c.metric === "mae");
  const [v2, sum, max] = [mae("countonce_v2"), mae("sum"), mae("max")];
  return [
    {
      title: "Compared with adding up per-photo counts",
      body: "Counting each photo and summing double-counts every object seen twice.",
      evidence:
        v2 && sum && max
          ? `On real apple trees with ${PHOTOS} photos, the mean count error in apples was ${v2.estimate.toFixed(1)} for CountOnce v2, ${sum.estimate.toFixed(1)} for the sum and ${max.estimate.toFixed(1)} for the largest single-photo count (lower is better). These errors are large.`
          : null,
    },
    {
      title: "Ordinary photos, no depth sensor",
      body: "The 3D scene is estimated from the photos themselves.",
      evidence: null,
    },
    {
      title: "Benchmark with confidence intervals",
      body: "Methods are compared on public data, including where CountOnce does worse.",
      evidence: v2 ? `${v2.n} held-out test trees, with 95% bootstrap intervals.` : null,
    },
  ];
}

const STEPS = [
  ["Detect", "An open-vocabulary detector finds your target in every photo."],
  ["Lift to 3D", "A feed-forward model places each detection in a shared 3D scene."],
  ["Merge", "Detections that are the same physical object are merged, with the reason recorded."],
  ["Count", "You get one number, a range showing how sensitive it is to settings, and warnings."],
];

export default function Home() {
  const scenes = listDemoScenes();
  const hero = scenes[0];
  const points = proofPoints(loadFuji());
  const rise = (i: number) => ({ animationDelay: `${i * 80}ms` });
  const heading = "text-2xl font-semibold tracking-tight";
  return (
    <PageShell obsidian>
      <MotionProvider>
        <section className="mx-auto grid max-w-6xl gap-10 px-4 py-14 lg:grid-cols-[1fr_1.05fr] lg:items-center lg:py-20">
          <div>
            <p className="rise surface mb-5 inline-flex rounded-full px-3 py-1 text-xs text-muted-foreground" style={rise(0)}>
              Research prototype
            </p>
            <h1 className="rise text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl" style={rise(1)}>
              Count anything from a few photos, without counting it twice.
            </h1>
            <p className="rise mt-5 text-lg text-muted-foreground" style={rise(2)}>
              CountOnce merges what every photo saw into one 3D scene, so an object that appears in
              five photos is counted once.
            </p>
            <p className="rise mt-2 text-sm text-muted-foreground" style={rise(3)}>
              It is not yet accurate on real photos.
            </p>
            <div className="rise mt-7 flex flex-wrap gap-3" style={rise(4)}>
              <Button asChild size="lg">
                <Link href={hero ? `/demo/${hero.id}` : "/demo"}>Explore the interactive demo</Link>
              </Button>
              <Button size="lg" variant="outline" disabled>
                Count your own photos: coming soon
              </Button>
            </div>
          </div>
          <HeroDiagram />
        </section>

        <section aria-labelledby="how" className="mx-auto max-w-6xl px-4 py-10">
          <h2 id="how" className={heading}>
            How it works
          </h2>
          <ol className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {STEPS.map(([title, body], i) => (
              <li key={title}>
                <SpotlightCard>
                  <p className="tabular text-xs text-muted-foreground">Step {i + 1}</p>
                  <h3 className="mt-1 font-semibold">{title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{body}</p>
                </SpotlightCard>
              </li>
            ))}
          </ol>
        </section>

        <section aria-labelledby="proof" className="mx-auto max-w-6xl px-4 py-10">
          <h2 id="proof" className={heading}>
            What we claim, and the evidence
          </h2>
          <p className="mt-1 text-sm">
            <Link href="/benchmark" className="underline underline-offset-4">
              See the full benchmark
            </Link>
          </p>
          <ul className="mt-5 grid gap-4 md:grid-cols-3">
            {points.map((p) => (
              <li key={p.title}>
                <SpotlightCard>
                  <h3 className="font-semibold">{p.title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{p.body}</p>
                  {p.evidence ? (
                    <p className="tabular mt-3 border-t border-border pt-3 text-sm">{p.evidence}</p>
                  ) : (
                    <p className="mt-3 rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
                      No published number yet.
                    </p>
                  )}
                </SpotlightCard>
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="gallery" className="mx-auto max-w-6xl px-4 py-10">
          <div className="flex items-baseline justify-between">
            <h2 id="gallery" className={heading}>
              Demo gallery
            </h2>
            <Link href="/demo" className="text-sm underline underline-offset-4">
              See all
            </Link>
          </div>
          <ul className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {scenes.map((s) => (
              <li key={s.id}>
                <DemoSceneCard scene={s} />
              </li>
            ))}
          </ul>
        </section>
      </MotionProvider>
    </PageShell>
  );
}
