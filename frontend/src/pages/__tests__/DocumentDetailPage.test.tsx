import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import DocumentDetailPage from "@/pages/DocumentDetailPage";
import { documentsApi } from "@/services/api";

vi.mock("@/services/api", () => ({
  documentsApi: { get: vi.fn(), progress: vi.fn(), reprocess: vi.fn(), cancel: vi.fn() },
  fileUrl: (p: string) => p,
  wsUrlForDocument: (id: string) => `ws://test/ws/documents/${id}`,
}));

function doc(status: string) {
  return {
    id: "doc-1",
    original_filename: "book.pdf",
    status,
    page_count: 3,
    dpi: 150,
    ocr_provider: "paddleocr",
    preprocess_profile: "FAST",
    error_message: status === "FAILED" ? "Processing stalled" : null,
  };
}

function job(status: string) {
  return {
    id: "job-1",
    document_id: "doc-1",
    status,
    stage: "ocr",
    progress_percent: 65,
    started_at: "2026-09-29T10:00:00Z",
    finished_at: "2026-09-29T11:00:00Z",
    processing_duration_seconds: 3600,
    error_message: null,
  };
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/documents/doc-1"]}>
        <Routes>
          <Route path="/documents/:id" element={<DocumentDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("DocumentDetailPage retry", () => {
  beforeEach(() => {
    vi.mocked(documentsApi.get).mockReset();
    vi.mocked(documentsApi.progress).mockReset();
    vi.mocked(documentsApi.reprocess).mockReset();
  });

  it("offers Retry on a failed document and re-runs it through the existing process endpoint", async () => {
    vi.mocked(documentsApi.get).mockResolvedValue(doc("FAILED") as never);
    vi.mocked(documentsApi.progress).mockResolvedValue(job("FAILED") as never);
    vi.mocked(documentsApi.reprocess).mockResolvedValue(job("QUEUED") as never);
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));

    await waitFor(() => expect(documentsApi.reprocess).toHaveBeenCalledWith("doc-1"));
    await waitFor(() => expect(vi.mocked(documentsApi.get).mock.calls.length).toBeGreaterThan(1));
  });

  it("does not offer Retry on a completed document", async () => {
    vi.mocked(documentsApi.get).mockResolvedValue(doc("COMPLETED") as never);
    vi.mocked(documentsApi.progress).mockResolvedValue(job("COMPLETED") as never);
    renderPage();

    await screen.findByText("book.pdf");
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("does not re-request the job endpoint in a loop after it 404s", async () => {
    vi.mocked(documentsApi.get).mockResolvedValue(doc("FAILED") as never);
    vi.mocked(documentsApi.progress).mockRejectedValue(new Error("404"));
    renderPage();

    await screen.findByText("book.pdf");
    await new Promise((r) => setTimeout(r, 300));
    expect(vi.mocked(documentsApi.progress).mock.calls.length).toBe(1);
  });

  it("shows the job's persisted stage and percent before any live event arrives", async () => {
    vi.mocked(documentsApi.get).mockResolvedValue(doc("OCR_PROCESSING") as never);
    vi.mocked(documentsApi.progress).mockResolvedValue({ ...job("OCR_PROCESSING"), finished_at: null, processing_duration_seconds: null } as never);
    renderPage();

    expect(await screen.findByText("65%")).toBeTruthy();
    expect(screen.queryByText("0%")).toBeNull();
  });
});
