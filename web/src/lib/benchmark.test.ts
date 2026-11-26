// @vitest-environment node
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { loadFuji, loadSynth, parseCsv } from "./benchmark";

const dir = () => mkdtempSync(join(tmpdir(), "bench-"));

describe("benchmark CSVs", () => {
  it("returns null when the files are missing or hold only a header", () => {
    const d = dir();
    expect(loadFuji(d)).toBeNull();
    expect(loadSynth(d)).toBeNull();
    writeFileSync(join(d, "benchmark_summary.csv"), "dataset,method\n");
    expect(loadFuji(d)).toBeNull();
  });

  it("reads the summary and drops paired rows from the synthetic table", () => {
    const d = dir();
    writeFileSync(
      join(d, "benchmark_summary.csv"),
      "dataset,method,k,metric,estimate,low,high,n,protocol,git_commit,git_dirty,timestamp_utc\n" +
        "fuji_sfm,sum,3,mae,2.5,1,4,8,p1,abc,true,2026-01-02T03:04:05+00:00\n",
    );
    writeFileSync(join(d, "v3_capped_synth_eval.csv"), "method,metric,mean,lo,hi,n,p\nv1,count_error,3,2,4,10,\nv3-minus-v1,count_error,-1,-2,0,10,0.0\n");
    expect(loadFuji(d)).toMatchObject({ commit: "abc", dirty: true, date: "2026-01-02", cells: [{ method: "sum", k: 3, estimate: 2.5, low: 1, high: 4, n: 8 }] });
    expect(loadSynth(d)).toHaveLength(1);
  });

  it("rejects malformed rows instead of showing wrong numbers", () => {
    expect(() => parseCsv("a,b\n1\n")).toThrow();
    expect(() => parseCsv('a\n"x"\n')).toThrow();
  });
});
