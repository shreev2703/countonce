import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge } from "./status-badge";

describe("StatusBadge", () => {
  it("labels each status and uses a different outline style, not colour alone", () => {
    const { rerender } = render(<StatusBadge status="new" />);
    expect(screen.getByText("Counted").className).toContain("border-solid");
    rerender(<StatusBadge status="duplicate" />);
    expect(screen.getByText("Duplicate").className).toContain("border-dashed");
    rerender(<StatusBadge status="rejected" />);
    expect(screen.getByText("Rejected").className).toContain("border-dotted");
  });
});
