// Copies the public benchmark summary into web/content/benchmark so a deploy rooted at web/ has it.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "..", "results", "benchmark_summary.csv");
if (!existsSync(src)) {
  console.log("sync-results: ../results/benchmark_summary.csv not found, keeping the committed copy");
} else {
  mkdirSync(join(root, "content", "benchmark"), { recursive: true });
  copyFileSync(src, join(root, "content", "benchmark", "benchmark_summary.csv"));
  console.log("sync-results: updated content/benchmark/benchmark_summary.csv");
}
