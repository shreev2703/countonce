import { describe, expect, it } from "vitest";

import { safeNext } from "./safe-next";

describe("safeNext", () => {
  it("keeps same-site paths", () => {
    expect(safeNext("/jobs/abc?x=1#y")).toBe("/jobs/abc?x=1#y");
  });
  it.each(["//evil.com", "/\\evil.com", "/\t/evil.com", "/.//evil.com", "https://evil.com", "evil", "", null, undefined])(
    "rejects %j",
    (v) => {
      expect(safeNext(v)).toBe("/new");
    },
  );
});
