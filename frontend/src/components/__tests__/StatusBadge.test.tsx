import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import StatusBadge from "@/components/StatusBadge";

describe("StatusBadge", () => {
  it("renders a human-readable label for underscored statuses", () => {
    render(<StatusBadge status="OCR_PROCESSING" />);
    expect(screen.getByText("OCR PROCESSING")).toBeInTheDocument();
  });

  it("renders COMPLETED with success styling", () => {
    render(<StatusBadge status="COMPLETED" />);
    const badge = screen.getByText("COMPLETED");
    expect(badge.className).toContain("emerald");
  });

  it("renders FAILED with error styling", () => {
    render(<StatusBadge status="FAILED" />);
    const badge = screen.getByText("FAILED");
    expect(badge.className).toContain("red");
  });
});
