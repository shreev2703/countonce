/**
 * Build check (PLAN 0.1 rule 3): no placeholder marker in anything production serves. Scans the
 * prerendered pages, the browser bundles, public files, and the source of server-rendered pages
 * (those have no prerendered output to scan). Runs as `postbuild`, so a hit fails `npm run build`.
 */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const MARKER = /\bTODO\b/;

function walk(dir, keep, out = []) {
  if (!existsSync(dir)) return out;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) walk(p, keep, out);
    else if (keep.test(p)) out.push(p);
  }
  return out;
}

export function findTodos(root) {
  const text = /\.(html|rsc|body|meta|js|css|json|txt|svg|md)$/;
  const files = [
    ...walk(join(root, ".next", "server", "app"), /\.(html|rsc|body|meta)$/),
    ...walk(join(root, ".next", "static"), text),
    ...walk(join(root, "public"), text),
    ...walk(join(root, "src"), /^(?!.*\.test\.).*\.(ts|tsx|css)$/),
  ];
  return { scanned: files.length, hits: files.filter((f) => MARKER.test(readFileSync(f, "utf8"))) };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const root = process.argv[2] ?? join(fileURLToPath(import.meta.url), "..", "..");
  if (!existsSync(join(root, ".next", "server", "app"))) {
    console.error("check-no-todo: no build output; run `npm run build` first");
    process.exit(1);
  }
  const { scanned, hits } = findTodos(root);
  if (hits.length) {
    console.error(`check-no-todo: placeholder marker found in:\n${hits.join("\n")}`);
    process.exit(1);
  }
  console.log(`check-no-todo: ${scanned} files clean`);
}
