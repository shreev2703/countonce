import { createHash, randomBytes } from "node:crypto";

import { describe, expect, it } from "vitest";

import { canSubmit, runPreflight, type Env, type FileInfo } from "./preflight";
import { clearUploadState, sessionKey, uploadFile, type KeyValueStore } from "./resumable";
import { Sha256, sha256Blob, sha256Hex } from "./sha256";

const ref = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");

describe("sha256", () => {
  it.each([0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 100_000])("matches node for %i bytes", (n) => {
    const b = new Uint8Array(randomBytes(n));
    expect(sha256Hex(b)).toBe(ref(b));
  });
  it("is the same when fed in odd pieces and as a Blob", async () => {
    const b = new Uint8Array(randomBytes(70_001));
    const h = new Sha256();
    for (let i = 0; i < b.length; i += 777) h.update(b.subarray(i, i + 777));
    expect(h.hex()).toBe(ref(b));
    expect(await sha256Blob(new Blob([b]), 4096)).toBe(ref(b));
  });
});

class Mem implements KeyValueStore {
  m = new Map<string, string>();
  get length() {
    return this.m.size;
  }
  key(i: number) {
    return [...this.m.keys()][i] ?? null;
  }
  getItem(k: string) {
    return this.m.get(k) ?? null;
  }
  setItem(k: string, v: string) {
    this.m.set(k, v);
  }
  removeItem(k: string) {
    this.m.delete(k);
  }
}

/** A fake of the upload endpoints that can drop chunk requests on demand. */
function fakeServer(opts: { failChunks?: Map<number, number>; chunkStatus?: number } = {}) {
  const received = new Map<number, string>();
  const log: string[] = [];
  let finalized = false;
  const api = async (path: string, init?: RequestInit): Promise<Response> => {
    const method = init?.method ?? "GET";
    log.push(`${method} ${path}`);
    const j = (o: unknown, status = 200) => new Response(JSON.stringify(o), { status });
    if (path === "/uploads" && method === "POST") return j({ session_id: "s1" }, 201);
    const chunk = /^\/uploads\/s1\/chunks\/(\d+)$/.exec(path);
    if (chunk) {
      const i = Number(chunk[1]);
      const left = opts.failChunks?.get(i) ?? 0;
      if (left > 0) {
        opts.failChunks!.set(i, left - 1);
        throw new TypeError("network down");
      }
      if (opts.chunkStatus) return j({ error: { code: "ERR_CHUNK_CORRUPT", message: "bad chunk" } }, opts.chunkStatus);
      const digest = (init!.headers as Record<string, string>)["x-chunk-sha256"]!;
      received.set(i, digest);
      return j({ ok: true });
    }
    if (path === "/uploads/s1/finalize") {
      finalized = true;
      return j({ job_id: "j1" });
    }
    if (path === "/uploads/s1") return j({ status: finalized ? "finalized" : "open", received: [...received.keys()] });
    return j({}, 404);
  };
  return { api, received, log, isFinalized: () => finalized };
}

const file = (bytes: Uint8Array, name = "a.jpg") => {
  const blob = new Blob([bytes as BlobPart]);
  return { name, size: blob.size, lastModified: 1, slice: (a: number, b: number) => blob.slice(a, b) };
};
const noSleep = async () => undefined;

describe("uploadFile", () => {
  it("sends every chunk with its hash and finalizes", async () => {
    const data = new Uint8Array(randomBytes(2500));
    const srv = fakeServer();
    const store = new Mem();
    const prog: number[] = [];
    await uploadFile(file(data), { api: srv.api, store, jobId: "j1", chunkBytes: 1000, onProgress: (s) => prog.push(s) });
    expect([...srv.received.keys()].sort()).toEqual([0, 1, 2]);
    expect(srv.received.get(1)).toBe(ref(data.subarray(1000, 2000)));
    expect(srv.isFinalized()).toBe(true);
    expect(prog.at(-1)).toBe(2500);
  });

  it("retries a dropped chunk with backoff", async () => {
    const srv = fakeServer({ failChunks: new Map([[1, 2]]) });
    await uploadFile(file(new Uint8Array(randomBytes(2500))), { api: srv.api, store: new Mem(), jobId: "j1", chunkBytes: 1000, sleep: noSleep });
    expect(srv.log.filter((l) => l.endsWith("/chunks/1")).length).toBe(3);
    expect(srv.isFinalized()).toBe(true);
  });

  it("does not retry a permanent 4xx", async () => {
    const srv = fakeServer({ chunkStatus: 422 });
    await expect(uploadFile(file(new Uint8Array(10)), { api: srv.api, store: new Mem(), jobId: "j1", chunkBytes: 4, sleep: noSleep })).rejects.toMatchObject({
      code: "ERR_CHUNK_CORRUPT",
      retryable: false,
    });
    expect(srv.log.filter((l) => l.includes("/chunks/0")).length).toBe(1);
  });

  it("resumes after a failure and only sends the missing chunks", async () => {
    const data = new Uint8Array(randomBytes(3500));
    const fail = new Map([[2, 99]]);
    const srv = fakeServer({ failChunks: fail });
    const store = new Mem();
    const f = file(data);
    const o = { api: srv.api, store, jobId: "j1", chunkBytes: 1000, retries: 1, sleep: noSleep };
    await expect(uploadFile(f, o)).rejects.toThrow();
    expect([...srv.received.keys()].sort()).toEqual([0, 1]);
    expect(store.getItem(sessionKey("j1", f))).not.toBeNull();

    fail.clear(); // network is back; a fresh call (page reload) picks up the stored session
    srv.log.length = 0;
    const out = await uploadFile(f, o);
    expect(out).toEqual({ resumed: true, skipped: 2 });
    expect(srv.log.filter((l) => l.startsWith("PUT"))).toEqual(["PUT /uploads/s1/chunks/2", "PUT /uploads/s1/chunks/3"]);
    expect(srv.log).not.toContain("POST /uploads");
    expect(srv.isFinalized()).toBe(true);

    srv.log.length = 0;
    await uploadFile(f, o); // retrying the whole submit must not upload it again
    expect(srv.log).toEqual([]);
  });
});

describe("preflight", () => {
  const env: Env = { online: true, hasCrypto: true, hasSlice: true };
  const photos = (n: number, extra: Partial<FileInfo> = {}): FileInfo[] =>
    Array.from({ length: n }, (_, i) => ({ name: `p${i}.jpg`, size: 2_000_000, width: 4000, height: 3000, ...extra }));
  const level = (c: ReturnType<typeof runPreflight>, id: string) => c.find((x) => x.id === id)?.level;

  it("passes 20 good photos", () => {
    const c = runPreflight(photos(20), env);
    expect(canSubmit(c)).toBe(true);
    expect(c.every((x) => x.level === "pass")).toBe(true);
  });
  it("fails with no files and with too few photos", () => {
    expect(canSubmit(runPreflight([], env))).toBe(false);
    expect(level(runPreflight(photos(3), env), "count")).toBe("fail");
    expect(level(runPreflight(photos(8), env), "count")).toBe("warn");
  });
  it("accepts one video instead of photos", () => {
    const c = runPreflight([{ name: "walk.mp4", size: 50_000_000 }], env);
    expect(canSubmit(c)).toBe(true);
  });
  it("rejects HEIC, oversize and corrupt photos with a fix", () => {
    const c = runPreflight([...photos(8), { name: "x.heic", size: 1 }], env);
    expect(level(c, "types")).toBe("fail");
    expect(c.find((x) => x.id === "types")!.detail).toMatch(/HEIC/);
    expect(level(runPreflight(photos(8, { size: 60 * 1024 * 1024 }), env), "size")).toBe("fail");
    expect(level(runPreflight(photos(8, { width: 0, height: 0 }), env), "decode")).toBe("fail");
  });
  it("warns on small photos, duplicates and offline; fails without crypto", () => {
    expect(level(runPreflight(photos(8, { width: 400, height: 300 }), env), "resolution")).toBe("warn");
    expect(level(runPreflight([...photos(8), photos(1)[0]!], env), "duplicates")).toBe("warn");
    expect(level(runPreflight(photos(8), { ...env, online: false }), "network")).toBe("warn");
    expect(canSubmit(runPreflight(photos(8), { ...env, hasCrypto: false }))).toBe(false);
  });
});

describe("clearUploadState", () => {
  it("drops only the given job's entries", () => {
    const s = new Mem();
    s.setItem("co:upload:j1:a:1:1", "x");
    s.setItem("co:upload:j2:a:1:1", "x");
    clearUploadState(s, "j1");
    expect(s.length).toBe(1);
  });
});
