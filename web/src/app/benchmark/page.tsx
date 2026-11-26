import type { Metadata } from "next";

import { BenchmarkExplorer } from "@/components/benchmark/explorer";
import { PageShell } from "@/components/chrome/site-chrome";
import { loadFuji, loadSynth, type FujiSummary, type SynthRow } from "@/lib/benchmark";

export const metadata: Metadata = {
  title: "Benchmark · CountOnce",
  description: "Measured results for CountOnce and simple per-photo baselines, with confidence intervals.",
};

export const dynamic = "force-static";

function Empty() {
  return (
    <p data-testid="benchmark-empty" className="mt-4 rounded-lg border border-dashed border-border p-6 text-muted-foreground">
      Benchmark in progress. Numbers appear here once a run has produced them.
    </p>
  );
}

function SliceWins({ fuji }: { fuji: FujiSummary }) {
  const wins = fuji.cells.filter((c) => c.metric.startsWith("slices_won_vs_"));
  if (wins.length === 0) return null;
  const ks = [...new Set(wins.map((c) => c.k))].sort((a, b) => a - b);
  const rows = ["countonce_v1", "countonce_v2"].flatMap((m) =>
    ["sum", "max"].map((b) => ({ m, b, label: `${m === "countonce_v1" ? "v1" : "v2"} beats ${b === "sum" ? "Sum" : "Max"}` })),
  );
  return (
    <>
      <h3 className="mt-8 text-lg font-semibold">Where merging in 3D helps, and where it does not</h3>
      <p className="mt-1 text-sm text-muted-foreground">
        Number of test slices (one side of one tree) where CountOnce had a strictly smaller error than the baseline.
      </p>
      <div className="mt-2 overflow-x-auto" tabIndex={0} role="region" aria-label="Slices won">
        <table className="tabular w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left">
              <th scope="col" className="py-2 pr-4">Comparison</th>
              {ks.map((k) => (
                <th key={k} scope="col" className="py-2 pr-4">{k} photos</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} className="border-b border-border">
                <th scope="row" className="py-2 pr-4 text-left font-medium">{r.label}</th>
                {ks.map((k) => {
                  const c = wins.find((w) => w.method === r.m && w.k === k && w.metric === `slices_won_vs_${r.b}`);
                  return (
                    <td key={k} className="py-2 pr-4">{c ? `${c.estimate} of ${c.n}` : "not measured"}</td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

const SYNTH_METRICS = [
  ["count_error", "Absolute count error", 1],
  ["duplicate_rate", "Duplicate rate", 3],
  ["miss_rate", "Miss rate", 3],
  ["assoc_f1", "Association F1", 3],
] as const;

function SynthTable({ rows }: { rows: SynthRow[] }) {
  const methods = [...new Set(rows.map((r) => r.method))];
  return (
    <div className="mt-3 overflow-x-auto" tabIndex={0} role="region" aria-label="Synthetic component test values">
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left">
            <th scope="col" className="py-2 pr-4">Matcher</th>
            {SYNTH_METRICS.map(([id, label]) => (
              <th key={id} scope="col" className="py-2 pr-4">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {methods.map((m) => (
            <tr key={m} className="border-b border-border">
              <th scope="row" className="py-2 pr-4 text-left font-medium">{m}</th>
              {SYNTH_METRICS.map(([id, , digits]) => {
                const r = rows.find((x) => x.method === m && x.metric === id);
                return (
                  <td key={id} className="whitespace-nowrap py-2 pr-4">
                    {r ? `${r.mean.toFixed(digits)} [${r.lo.toFixed(digits)}, ${r.hi.toFixed(digits)}]` : "not measured"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function BenchmarkPage() {
  const fuji = loadFuji();
  const synth = loadSynth();
  return (
    <PageShell>
      <div className="mx-auto max-w-5xl px-4 py-10">
        <h1 className="text-3xl font-semibold">Benchmark</h1>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          CountOnce against the obvious shortcuts: adding up per-photo counts (Sum), taking the largest (Max) or the
          middle one (Median). Results are shown as measured, including where CountOnce does worse.
        </p>
        {!fuji && !synth && <Empty />}

        {(fuji || synth) && (
          <section aria-labelledby="fuji" className="mt-10">
            <h2 id="fuji" className="text-2xl font-semibold">Apples on real trees (Fuji-SfM)</h2>
            {fuji ? (
              <>
                <p className="mb-4 mt-2 max-w-2xl text-sm text-muted-foreground">
                  Held-out test trees of the public Fuji-SfM dataset (research use, CC BY-NC-SA; no images shown), with settings frozen on other trees first. Errors
                  are large, and no method here is accurate enough to rely on.
                </p>
                <BenchmarkExplorer cells={fuji.cells} />
                <SliceWins fuji={fuji} />
                <p className="mt-4 text-xs text-muted-foreground">
                  Protocol {fuji.protocol}. Runs dated {fuji.date} at commit {fuji.commit}
                  {fuji.dirty ? " with uncommitted changes in the working tree" : ""}.
                </p>
              </>
            ) : (
              <Empty />
            )}
          </section>
        )}

        {(fuji || synth) && (
          <section aria-labelledby="synth" className="mt-12">
            <h2 id="synth" className="text-2xl font-semibold">Matcher component test (synthetic only)</h2>
            {synth ? (
              <>
                <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
                  Generated scenes, testing only the step that decides which detections are the same object. v3 is a
                  learned matcher trained on synthetic data; v1 and v2 were not tuned for this test, and the test
                  has no occlusion check. It says nothing about accuracy on real photos. {synth[0]?.n} test episodes, mean
                  [95% interval].
                </p>
                <SynthTable rows={synth} />
              </>
            ) : (
              <Empty />
            )}
          </section>
        )}

        <section aria-labelledby="missing" className="mt-12">
          <h2 id="missing" className="text-2xl font-semibold">Not measured yet</h2>
          <ul className="mt-2 list-disc pl-5 text-sm text-muted-foreground">
            <li>Our own captured scenes, including scenes with none of the target object.</li>
            <li>Ordered against shuffled photos, and results by scene type.</li>
            <li>A vision-language model baseline.</li>
            <li>A failure gallery and a public leaderboard.</li>
          </ul>
        </section>
      </div>
    </PageShell>
  );
}
