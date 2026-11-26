/**
 * Build-time reader of the benchmark CSVs in `results/` (server components only; PLAN 17.8).
 * A missing or empty file gives `null`, which the page renders as an empty state.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export interface FujiCell {
  method: string;
  k: number;
  metric: string;
  estimate: number;
  low: number | null;
  high: number | null;
  n: number;
}

export interface FujiSummary {
  cells: FujiCell[];
  protocol: string;
  commit: string;
  dirty: boolean;
  date: string;
}

export interface SynthRow {
  method: string;
  metric: string;
  mean: number;
  lo: number;
  hi: number;
  n: number;
}

/**
 * Where the result CSVs live: `COUNTONCE_RESULTS_DIR`, else the repo's `../results` (local builds),
 * else the committed copy in `content/benchmark` (a Vercel project rooted at `web/` cannot see
 * `../results`). `npm run sync:results` refreshes the committed copy.
 */
export function resultsDir(): string {
  if (process.env.COUNTONCE_RESULTS_DIR) return process.env.COUNTONCE_RESULTS_DIR;
  const repo = join(process.cwd(), "..", "results");
  return existsSync(join(repo, "benchmark_summary.csv")) ? repo : join(process.cwd(), "content", "benchmark");
}

/** The result CSVs read here hold no quoted fields; a quote means the format changed. */
export function parseCsv(text: string): Record<string, string>[] {
  const [head, ...lines] = text.split(/\r?\n/).filter((l) => l !== "");
  if (!head) return [];
  if (text.includes('"')) throw new Error("quoted CSV fields are not supported");
  const cols = head.split(",");
  return lines.map((line) => {
    const cells = line.split(",");
    if (cells.length !== cols.length) throw new Error(`bad CSV row: ${line}`);
    return Object.fromEntries(cols.map((c, i) => [c, cells[i] ?? ""]));
  });
}

function read(dir: string, name: string): Record<string, string>[] {
  const path = join(/*turbopackIgnore: true*/ dir, name);
  return existsSync(path) ? parseCsv(readFileSync(path, "utf8")) : [];
}

function num(s: string | undefined, what: string): number {
  const v = Number(s);
  if (s === undefined || s === "" || !Number.isFinite(v)) throw new Error(`bad number in ${what}: ${s}`);
  return v;
}

export function loadFuji(dir = resultsDir()): FujiSummary | null {
  const rows = read(dir, "benchmark_summary.csv").filter((r) => r.dataset === "fuji_sfm");
  const first = rows[0];
  if (!first) return null;
  return {
    protocol: first.protocol ?? "",
    commit: (first.git_commit ?? "").replaceAll(";", ", "),
    dirty: first.git_dirty === "true",
    date: (first.timestamp_utc ?? "").slice(0, 10),
    cells: rows.map((r) => ({
      method: r.method ?? "",
      k: num(r.k, "k"),
      metric: r.metric ?? "",
      estimate: num(r.estimate, "estimate"),
      low: r.low ? num(r.low, "low") : null,
      high: r.high ? num(r.high, "high") : null,
      n: num(r.n, "n"),
    })),
  };
}

/** Synthetic association component test; paired-difference rows are left out. */
export function loadSynth(dir = resultsDir()): SynthRow[] | null {
  const rows = read(dir, "v3_capped_synth_eval.csv").filter((r) => !r.method?.includes("-minus-"));
  if (rows.length === 0) return null;
  return rows.map((r) => ({
    method: r.method ?? "",
    metric: r.metric ?? "",
    mean: num(r.mean, "mean"),
    lo: num(r.lo, "lo"),
    hi: num(r.hi, "hi"),
    n: num(r.n, "n"),
  }));
}
