/**
 * In-memory stand-in for the CountOnce API (same routes and error envelope), used by the
 * "mocked API" Playwright run. Test hooks live under /__mock:
 *   POST /__mock/reset
 *   POST /__mock/config  {workerOnline, sse, failChunks, resultCount}
 * Jobs advance on their own once submitted and a worker is "online".
 */
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { randomUUID } from "node:crypto";

const port = Number(process.env.MOCK_API_PORT ?? 4010);
const SECRET = process.env.COUNTONCE_AUTH_SECRET ?? "e2e-secret";
const fixture = JSON.parse(readFileSync(new URL("../../../tests/fixtures/schemas/v1.1/result.json", import.meta.url), "utf8"));
const STAGES = ["ingest", "detecting", "recon", "matching", "counting"];

let cfg, jobs, uploads, keys, shares, feedback, corrections, creates;
function reset() {
  cfg = { workerOnline: true, sse: true, failChunks: 0 };
  jobs = new Map();
  uploads = new Map();
  keys = new Map();
  shares = new Map();
  feedback = [];
  corrections = [];
  creates = 0;
}
reset();

const send = (res, status, body, headers = {}) => {
  res.writeHead(status, { "content-type": "application/json", ...headers });
  res.end(JSON.stringify(body));
};
const err = (res, status, code, message) => send(res, status, { error: { code, message, tip: null, retryable: status >= 500 } });
const view = (j) => ({
  id: j.id, prompt: j.prompt, status: j.status, stage: j.stage, progress: j.progress, error_code: j.error_code ?? null,
  error_message: j.error_message ?? null, attempts: 0, site_id: null, captured_at: null, created_at: j.created_at,
  started_at: null, finished_at: j.finished_at ?? null, negative_prompts: j.negative_prompts, options: j.options,
});

// A job moves one step every 150 ms while a worker is online.
setInterval(() => {
  if (!cfg.workerOnline) return;
  for (const j of jobs.values()) {
    if (j.status === "queued") Object.assign(j, { status: "running", stage: STAGES[0], progress: 0.1 });
    else if (j.status === "running") {
      const i = STAGES.indexOf(j.stage) + 1;
      if (i >= STAGES.length) Object.assign(j, { status: "done", stage: "done", progress: 1, finished_at: Date.now() / 1000 });
      else Object.assign(j, { stage: STAGES[i], progress: Math.min(0.95, 0.1 + i * 0.2) });
    }
  }
}, 150);

const body = (req) => new Promise((r) => { const c = []; req.on("data", (d) => c.push(d)); req.on("end", () => r(Buffer.concat(c))); });

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  const p = url.pathname;
  const m = req.method;
  const raw = await body(req);
  const json = () => (raw.length ? JSON.parse(raw.toString()) : {});

  if (p === "/__mock/reset") { reset(); return send(res, 200, {}); }
  if (p === "/__mock/config") { Object.assign(cfg, json()); return send(res, 200, cfg); }
  if (p === "/__mock/state") return send(res, 200, { creates, jobs: [...jobs.values()].map((j) => ({ id: j.id, files: j.files.length, status: j.status })), feedback, corrections });

  if (p === "/health") {
    return send(res, 200, { status: "ok", workers_online: cfg.workerOnline ? 1 : 0, processing_paused: !cfg.workerOnline, workers: [], queue: { queued: 0, running: 0 } });
  }
  if (p === "/auth/exchange" && m === "POST") {
    if (req.headers["x-auth-secret"] !== SECRET) return err(res, 401, "ERR_UNAUTHENTICATED", "bad auth secret");
    const { provider, subject } = json();
    const user = `${provider}:${subject}`;
    const key = `co_${randomUUID().slice(0, 12)}_${randomUUID()}`;
    keys.set(key, user);
    return send(res, 200, { user_id: user, api_key: key });
  }
  let share = /^\/share\/([^/]+)$/.exec(p);
  if (share && m === "GET") {
    const id = shares.get(share[1]);
    const j = id && jobs.get(id);
    if (!j) return err(res, 404, "ERR_NOT_FOUND", "unknown or revoked share link");
    return send(res, 200, { job: { id: j.id, status: j.status, stage: j.stage, progress: j.progress, prompt: j.prompt, created_at: j.created_at, finished_at: j.finished_at }, result: { result_json: resultFor(j) } });
  }

  const auth = req.headers.authorization?.replace(/^Bearer /i, "");
  const user = keys.get(auth);
  if (!user) return err(res, 401, "ERR_UNAUTHENTICATED", "missing or invalid API key");
  const own = (id) => { const j = jobs.get(id); return j && j.user === user ? j : null; };

  if (p === "/jobs" && m === "POST") {
    const b = json();
    const ik = req.headers["idempotency-key"];
    const prior = ik && [...jobs.values()].find((j) => j.user === user && j.ik === ik);
    if (prior) return send(res, 200, { id: prior.id, status: prior.status, idempotent_replay: true });
    creates++;
    const j = { id: randomUUID().replaceAll("-", ""), user, ik, prompt: b.prompt, negative_prompts: b.negative_prompts ?? [], options: b.options ?? {}, status: "draft", stage: "draft", progress: 0, created_at: Date.now() / 1000, files: [] };
    jobs.set(j.id, j);
    return send(res, 201, { id: j.id, status: j.status, idempotent_replay: false });
  }
  if (p === "/jobs" && m === "GET") return send(res, 200, { jobs: [...jobs.values()].filter((j) => j.user === user).map(view) });

  if (p === "/uploads" && m === "POST") {
    const b = json();
    if (!/\.(png|jpe?g|webp|bmp|mp4|mov)$/i.test(b.filename)) return err(res, 415, "ERR_UNSUPPORTED_MEDIA", "unsupported file type");
    const id = randomUUID().replaceAll("-", "");
    uploads.set(id, { user, n: Math.ceil(b.total_bytes / b.chunk_size), got: new Set(), filename: b.filename, status: "open" });
    return send(res, 201, { session_id: id, n_chunks: uploads.get(id).n });
  }
  const up = /^\/uploads\/([^/]+)(?:\/(chunks\/(\d+)|finalize))?$/.exec(p);
  if (up) {
    const s = uploads.get(up[1]);
    if (!s || s.user !== user) return err(res, 404, "ERR_NOT_FOUND", "no such upload session");
    if (!up[2] && m === "GET") return send(res, 200, { session_id: up[1], status: s.status, received: [...s.got], n_chunks: s.n, complete: s.got.size === s.n });
    if (up[3] !== undefined && m === "PUT") {
      if (cfg.failChunks > 0) { cfg.failChunks--; return err(res, 503, "ERR_STORAGE_TRANSIENT", "storage hiccup"); }
      s.got.add(Number(up[3]));
      return send(res, 200, { index: Number(up[3]) });
    }
    if (up[2] === "finalize" && m === "POST") {
      if (s.got.size !== s.n) return err(res, 409, "ERR_UPLOAD_INCOMPLETE", "chunks missing");
      s.status = "finalized";
      const j = own(json().job_id);
      if (j) j.files.push(s.filename);
      return send(res, 200, { session_id: up[1], job_id: j?.id ?? null });
    }
  }

  const jm = /^\/jobs\/([^/]+)(?:\/([a-z]+))?$/.exec(p);
  if (jm) {
    const j = own(jm[1]);
    if (!j) return err(res, 404, "ERR_NOT_FOUND", `no such job: ${jm[1]}`);
    const sub = jm[2];
    if (!sub && m === "GET") return send(res, 200, { ...view(j), files: j.files.length });
    if (!sub && m === "DELETE") { jobs.delete(j.id); return send(res, 200, { id: j.id, status: "deleted" }); }
    if (sub === "submit" && m === "POST") {
      if (j.status === "draft") {
        if (!j.files.length) return err(res, 400, "ERR_INVALID_INPUT", "the job has no uploaded files");
        Object.assign(j, { status: "queued", stage: "queued" });
      }
      return send(res, 200, { ...view(j), idempotent_replay: false });
    }
    if (sub === "events" && m === "GET") {
      if (!cfg.sse) return err(res, 503, "ERR_STORAGE_TRANSIENT", "events unavailable");
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
      let last = "";
      const t = setInterval(() => {
        const state = JSON.stringify([j.status, j.stage, j.progress]);
        if (state !== last) {
          last = state;
          const done = ["done", "failed"].includes(j.status);
          res.write(`event: ${done ? "done" : "progress"}\ndata: ${JSON.stringify(view(j))}\n\n`);
          if (done) { clearInterval(t); res.end(); }
        } else res.write(": keep-alive\n\n");
      }, 100);
      req.on("close", () => clearInterval(t));
      return;
    }
    if (sub === "result" && m === "GET") {
      if (j.status !== "done") return err(res, 409, "ERR_NOT_READY", `job is ${j.status}`);
      return send(res, 200, { job: view(j), result: { result_json: resultFor(j), warnings_json: [] }, outputs: [] });
    }
    if (sub === "share" && m === "POST") {
      const token = `st_${randomUUID().replaceAll("-", "")}`;
      shares.set(token, j.id);
      return send(res, 201, { token, url: `/share/${token}` });
    }
    if (sub === "feedback" && m === "POST") { feedback.push(json()); return send(res, 201, { id: "f1" }); }
    if (sub === "corrections" && m === "GET") return send(res, 200, { corrections, corrected_count: corrections.length ? fixture.count - corrections.length : null });
    if (sub === "corrections" && m === "POST") {
      corrections.push(json());
      return send(res, 200, { id: "c1", idempotent_replay: false, corrected_count: fixture.count - corrections.length, corrections: corrections.length });
    }
  }
  return err(res, 404, "ERR_NOT_FOUND", `no route ${m} ${p}`);
}).listen(port, "127.0.0.1", () => console.log(`mock api on ${port}`));

function resultFor(j) {
  return { ...fixture, prompt: j.prompt };
}
