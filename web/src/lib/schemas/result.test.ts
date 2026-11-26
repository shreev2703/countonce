import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  CURRENT_VERSION,
  parseMarkers,
  parseResult,
  SchemaVersionError,
  SUPPORTED_VERSIONS,
} from "./result";

// Shared with the backend's tests/test_schema_fixtures.py: one stored example per version.
const ROOT = join(__dirname, "../../../../tests/fixtures/schemas");
const versions = readdirSync(ROOT).filter((d) => d.startsWith("v")).sort();
const load = (v: string, name: string): unknown => JSON.parse(readFileSync(join(ROOT, v, name), "utf8"));

describe("result.json / markers.json schemas", () => {
  it("has a fixture for every supported version, including the current one", () => {
    expect(versions.map((v) => v.slice(1))).toEqual([...SUPPORTED_VERSIONS]);
    expect(SUPPORTED_VERSIONS).toContain(CURRENT_VERSION);
  });

  it.each(versions)("%s result.json parses and upgrades to the current shape", (v) => {
    const raw = load(v, "result.json") as { count: number };
    const r = parseResult(raw);
    expect(r.schema_version).toBe(CURRENT_VERSION);
    expect(r.count).toBe(raw.count);
    expect(Array.isArray(r.detections)).toBe(true);
    expect(Array.isArray(r.cameras)).toBe(true);
  });

  it.each(versions)("%s markers.json parses", (v) => {
    const m = parseMarkers(load(v, "markers.json"));
    expect(m.schema_version).toBe(CURRENT_VERSION);
    expect(m.markers.length).toBeGreaterThanOrEqual(0);
  });

  it("rejects an unknown schema_version", () => {
    const raw = { ...(load(`v${CURRENT_VERSION}`, "result.json") as object), schema_version: "9.9" };
    expect(() => parseResult(raw)).toThrow(SchemaVersionError);
  });

  it("rejects a document without schema_version", () => {
    expect(() => parseResult({ count: 1 })).toThrow(SchemaVersionError);
  });

  it("is strict: unknown keys are an error", () => {
    const raw = { ...(load(`v${CURRENT_VERSION}`, "result.json") as object), surprise: 1 };
    expect(() => parseResult(raw)).toThrow();
  });
});
