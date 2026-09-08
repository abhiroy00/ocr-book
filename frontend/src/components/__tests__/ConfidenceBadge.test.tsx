import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ConfidenceBadge from "@/components/ConfidenceBadge";

describe("ConfidenceBadge", () => {
  it("shows a dash for null/undefined confidence", () => {
    render(<ConfidenceBadge value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("marks confidence above 95% as high (no warning icon)", () => {
    render(<ConfidenceBadge value={0.97} />);
    expect(screen.getByText("97%")).toBeInTheDocument();
  });

  it("flags confidence below 80% as needing review", () => {
    render(<ConfidenceBadge value={0.65} />);
    expect(screen.getByText("65% ⚠")).toBeInTheDocument();
  });

  it("treats 80-95% as a warning tier without the review flag", () => {
    render(<ConfidenceBadge value={0.85} />);
    expect(screen.getByText("85%")).toBeInTheDocument();
  });
});
