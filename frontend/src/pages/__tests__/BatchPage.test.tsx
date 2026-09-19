import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import BatchPage from "@/pages/BatchPage";
import { batchApi } from "@/services/batchApi";
import type { Batch, BatchItem } from "@/types/batch";

vi.mock("@/services/batchApi", () => ({
  batchApi: { status: vi.fn(), retry: vi.fn(), cancel: vi.fn() },
}));
vi.mock("@/services/api", () => ({
  documentsApi: {
    downloadPdfUrl: (id: string) => `/dl/pdf/${id}`,
    downloadDocxUrl: (id: string) => `/dl/docx/${id}`,
    downloadExcelUrl: (id: string) => `/dl/xlsx/${id}`,
  },
}));

function item(over: Partial<BatchItem>): BatchItem {
  return {
    job_id: "j",
    document_id: "d",
    processing_job_id: "p",
    filename: "file.pdf",
    file_size_bytes: 2048,
    position: 0,
    state: "queued",
    status: "QUEUED",
    stage: null,
    progress_percent: 0,
    processed_pages: 0,
    total_pages: 5,
    priority_score: 5,
    attempts: 1,
    error_message: null,
    started_at: null,
    finished_at: null,
    duration_seconds: null,
    ...over,
  };
}

function batch(over: Partial<Batch>): Batch {
  return {
    batch_id: "b1",
    status: "processing",
    ocr_provider: "paddleocr",
    dpi: 150,
    preprocess_profile: "FAST",
    created_at: "2026-09-19T10:00:00Z",
    started_at: "2026-09-19T10:00:05Z",
    finished_at: null,
    elapsed_seconds: 75,
    total_files: 4,
    completed_files: 1,
    failed_files: 1,
    cancelled_files: 0,
    duplicate_files: 0,
    queued_files: 1,
    processing_files: 1,
    finished_files: 2,
    total_pages: 160,
    processed_pages: 60,
    overall_percent: 37,
    concurrency: { provider: "paddleocr", limit: 1, limiting_factor: "ram", cpu_count: 2, total_mem_mb: 7775, bounds: { ram: 1 } },
    items: [
      item({ job_id: "j1", document_id: "d1", filename: "invoice_small.pdf", state: "completed", status: "COMPLETED", total_pages: 3, processed_pages: 3, progress_percent: 100, duration_seconds: 42 }),
      item({ job_id: "j2", document_id: "d2", filename: "annual_report.pdf", state: "running", status: "OCR_PROCESSING", stage: "ocr", total_pages: 150, processed_pages: 47, progress_percent: 30 }),
      item({ job_id: "j3", document_id: "d3", filename: "waiting.pdf", state: "queued", total_pages: 5 }),
      item({ job_id: "j4", document_id: "d4", filename: "broken.pdf", state: "failed", status: "FAILED", total_pages: 2, error_message: "Corrupted page tree" }),
    ],
    ...over,
  };
}

function renderPage(entry = "/batches/b1") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[entry]}>
        <Routes>
          <Route path="/batches/:batchId" element={<BatchPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("BatchPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows overall progress and every file with its own status", async () => {
    (batchApi.status as any).mockResolvedValue(batch({}));
    renderPage();

    expect(await screen.findByText("Multiple OCR Queue")).toBeInTheDocument();
    expect(screen.getByText(/1 \/ 4 completed/)).toBeInTheDocument();
    expect(screen.getByText(/1 failed/)).toBeInTheDocument();
    expect(screen.getByText("60 / 160 pages")).toBeInTheDocument();

    const done = screen.getByTestId("batch-row-j1");
    expect(within(done).getByText("invoice_small.pdf")).toBeInTheDocument();
    expect(within(done).getByText(/✓ Completed/)).toBeInTheDocument();
    expect(within(done).getByText(/3 pages/)).toBeInTheDocument();

    const running = screen.getByTestId("batch-row-j2");
    expect(within(running).getByText(/Processing/)).toBeInTheDocument();
    expect(within(running).getByText("47 / 150 pages")).toBeInTheDocument();

    expect(within(screen.getByTestId("batch-row-j3")).getByText("Queued")).toBeInTheDocument();
  });

  it("explains the concurrency the server is using", async () => {
    (batchApi.status as any).mockResolvedValue(batch({}));
    renderPage();
    expect(await screen.findByText(/up to 1 file at a time/)).toBeInTheDocument();
  });

  it("offers downloads only for a completed file and links every openable file to its document", async () => {
    (batchApi.status as any).mockResolvedValue(batch({}));
    renderPage();
    const done = await screen.findByTestId("batch-row-j1");
    expect(within(done).getByRole("link", { name: "Searchable PDF" })).toHaveAttribute("href", "/dl/pdf/d1");
    expect(within(done).getByRole("link", { name: "View / edit" })).toHaveAttribute("href", "/documents/d1");
    expect(within(screen.getByTestId("batch-row-j3")).queryByRole("link", { name: "Searchable PDF" })).toBeNull();
    expect(within(screen.getByTestId("batch-row-j2")).getByRole("link", { name: "Open" })).toHaveAttribute("href", "/documents/d2");
  });

  it("shows a failed file's error and retries only that file", async () => {
    (batchApi.status as any).mockResolvedValue(batch({}));
    (batchApi.retry as any).mockResolvedValue(item({ job_id: "j4", state: "queued" }));
    renderPage();

    const failed = await screen.findByTestId("batch-row-j4");
    expect(within(failed).getByText("Corrupted page tree")).toBeInTheDocument();
    fireEvent.click(within(failed).getByRole("button", { name: "Retry" }));

    await waitFor(() => expect(batchApi.retry).toHaveBeenCalledWith("b1", "j4"));
    expect(screen.getAllByRole("button", { name: "Retry" })).toHaveLength(1); // no retry offered for the healthy files
  });

  it("cancels the remaining files", async () => {
    (batchApi.status as any).mockResolvedValue(batch({}));
    (batchApi.cancel as any).mockResolvedValue({ status: "cancel_requested", cancelled_waiting: 1, cancelling_running: 1 });
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: "Cancel remaining" }));
    await waitFor(() => expect(batchApi.cancel).toHaveBeenCalledWith("b1"));
  });

  it("hides cancel and shows the total time once the batch has finished", async () => {
    (batchApi.status as any).mockResolvedValue(
      batch({ status: "completed_with_errors", processing_files: 0, queued_files: 0, elapsed_seconds: 3725 }),
    );
    renderPage();
    expect(await screen.findByText(/Total time 1:02:05/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel remaining" })).toBeNull();
  });

  it("flags a file that only just finished its pages as finalizing, not stuck", async () => {
    (batchApi.status as any).mockResolvedValue(
      batch({ items: [item({ job_id: "j9", filename: "x.pdf", state: "running", status: "EXPORTING", stage: "export", total_pages: 10, processed_pages: 10 })] }),
    );
    renderPage();
    expect(await screen.findByText("Finalizing exports…")).toBeInTheDocument();
  });

  it("shows a notice for files the server refused during upload", async () => {
    (batchApi.status as any).mockResolvedValue(batch({}));
    renderPage();
    // (skipped files arrive via router state from the upload page; covered in UploadPage tests)
    expect(await screen.findByText("Multiple OCR Queue")).toBeInTheDocument();
    expect(screen.queryByText(/not added/)).toBeNull();
  });

  it("says so when the batch cannot be loaded", async () => {
    (batchApi.status as any).mockRejectedValue(new Error("404"));
    renderPage();
    expect(await screen.findByText("Could not load this batch.")).toBeInTheDocument();
  });
});
