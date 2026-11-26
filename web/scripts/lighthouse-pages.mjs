/**
 * Lighthouse (v13, mobile preset) on the pages that need backend data: /jobs/[id] and /s/[token].
 * Starts the mocked API (e2e/support/mock-api.mjs) and `next start` on a built `.next`, seeds a finished job
 * plus a share token, signs in through the dev-auth route, runs each page N times and prints medians.
 * Usage (in web/, after `npm run build`): node scripts/lighthouse-pages.mjs [runs=3]
 */
import { spawn, execFileSync } from "node:child_process";
import { readFileSync, mkdirSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const runs = Number(process.argv[2] ?? 3);
const WEB = 3100, API = 4010, SECRET = "e2e-secret";
const env = { ...process.env, COUNTONCE_AUTH_SECRET: SECRET, COUNTONCE_SESSION_SECRET: "e2e-session-secret-not-for-prod", COUNTONCE_DEV_AUTH: "1", COUNTONCE_API_URL: `http://127.0.0.1:${API}`, MOCK_API_PORT: String(API) };
const kids = [spawn("node", ["e2e/support/mock-api.mjs"], { env, stdio: "ignore" }), spawn("npx", ["next", "start", "-p", String(WEB)], { env, stdio: "ignore" })];
const stop = () => kids.forEach((k) => k.kill());
process.on("exit", stop);

async function waitFor(url) {
  for (let i = 0; i < 60; i++) { try { if ((await fetch(url)).ok) return; } catch {} await sleep(500); }
  throw new Error(`timeout waiting for ${url}`);
}
await waitFor(`http://127.0.0.1:${API}/health`);
await waitFor(`http://127.0.0.1:${WEB}/`);

const api = (path, init = {}) => fetch(`http://127.0.0.1:${API}${path}`, init);
const jf = async (r) => { if (!r.ok) throw new Error(`${r.url} ${r.status}`); return r.json(); };
// Sign in through the dev-auth route; the mock maps the minted key to user "dev:lhuser".
const login = await fetch(`http://127.0.0.1:${WEB}/api/auth/dev`, { method: "POST", body: new URLSearchParams({ user: "lhuser", next: "/new" }), redirect: "manual" });
const cookie = login.headers.getSetCookie().map((c) => c.split(";")[0]).join("; ");
const { api_key } = await jf(await api("/auth/exchange", { method: "POST", headers: { "x-auth-secret": SECRET, "content-type": "application/json" }, body: JSON.stringify({ provider: "dev", subject: "lhuser" }) }));
const auth = { authorization: `Bearer ${api_key}`, "content-type": "application/json" };
const { id } = await jf(await api("/jobs", { method: "POST", headers: auth, body: JSON.stringify({ prompt: "red apples" }) }));
const up = await jf(await api("/uploads", { method: "POST", headers: auth, body: JSON.stringify({ filename: "a.png", total_bytes: 4, chunk_size: 4 }) }));
await api(`/uploads/${up.session_id}/chunks/0`, { method: "PUT", headers: auth, body: "abcd" });
await api(`/uploads/${up.session_id}/finalize`, { method: "POST", headers: auth, body: JSON.stringify({ job_id: id }) });
await api(`/jobs/${id}/submit`, { method: "POST", headers: auth });
for (let i = 0; i < 40; i++) { if ((await jf(await api(`/jobs/${id}`, { headers: auth }))).status === "done") break; await sleep(250); }
const { token } = await jf(await api(`/jobs/${id}/share`, { method: "POST", headers: auth }));

const chrome = execFileSync("node", ["-e", 'console.log(require("@playwright/test").chromium.executablePath())']).toString().trim();
mkdirSync("/tmp/lhrun", { recursive: true });
const pages = [[`/jobs/${id}`, { Cookie: cookie }], [`/s/${token}`, {}]];
let worst = 1;
for (const [path, headers] of pages) {
  const rows = [];
  for (let i = 0; i < runs; i++) {
    const out = "/tmp/lhrun/pages.json";
    execFileSync("npx", ["--yes", "lighthouse@13", `http://127.0.0.1:${WEB}${path}`, "--only-categories=performance,accessibility,best-practices", "--chrome-flags=--headless=new --no-sandbox", "--quiet", "--output=json", `--output-path=${out}`, `--extra-headers=${JSON.stringify(headers)}`], { cwd: "/tmp/lhrun", env: { ...process.env, CHROME_PATH: chrome }, stdio: "ignore" });
    const r = JSON.parse(readFileSync(out, "utf8"));
    const c = r.categories;
    rows.push({ perf: c.performance.score, a11y: c.accessibility.score, bp: c["best-practices"].score, lcp: r.audits["largest-contentful-paint"].displayValue, tbt: r.audits["total-blocking-time"].displayValue });
    if (new URL(r.finalDisplayedUrl).pathname !== path) console.warn(`WARNING: redirected to ${r.finalDisplayedUrl}`);
  }
  const med = (k) => rows.map((x) => x[k]).sort((a, b) => a - b)[Math.floor(rows.length / 2)];
  const m = { perf: med("perf"), a11y: med("a11y"), bp: med("bp") };
  worst = Math.min(worst, m.perf, m.a11y, m.bp);
  console.log(path.replace(id, "[id]").replace(token, "[token]"), JSON.stringify(rows), "median", JSON.stringify(m));
}
stop();
process.exit(worst >= 0.9 ? 0 : 1);
