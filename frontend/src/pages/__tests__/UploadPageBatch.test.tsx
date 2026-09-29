import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import UploadPage from "@/pages/UploadPage";
import { batchApi } from "@/services/batchApi";

vi.mock("@/services/api", () => ({
  documentsApi: { upload: vi.fn() },
  fileUrl: (p: string) => p,
}));
vi.mock("@/services/batchApi", () => ({
  batchApi: {
    create: vi.fn(),
    addFile: vi.fn(),
    start: vi.fn(),
    // UploadPage fetches this (for the per-file size limit) via react-query
    // on mount; a generous limit here means it never interferes with these
    // tests' own files, which are tiny.
    capacity: vi.fn().mockResolvedValue({
      concurrency: { provider: "paddleocr", limit: 2, limiting_factor: "cpu", cpu_count: 4, total_mem_mb: 8192, bounds: {} },
      max_files: 50,
      max_total_mb: 5000,
      max_file_mb: 1024,
      allowed_extensions: [".pdf", ".jpg", ".jpeg", ".png", ".webp"],
    }),
  },
}));

const mockedNavigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => mockedNavigate };
});

const pdf = (name: string) => new File(["%PDF-1.4"], name, { type: "application/pdf" });

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const result = render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { ...result, client };
}

function openMultiple() {
  fireEvent.click(screen.getByRole("tab", { name: "Multiple PDF OCR" }));
}

function pick(files: File[]) {
  fireEvent.change(screen.getByTestId("batch-file-input"), { target: { files } });
}

describe("UploadPage — Multiple PDF OCR", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (batchApi.create as any).mockResolvedValue({ batch_id: "b1" });
    (batchApi.start as any).mockResolvedValue({ batch_id: "b1", queued_files: 2, status: "processing" });
  });

  it("starts in single-file mode with the existing button and no multi-file input", () => {
    renderPage();
    expect(screen.getByRole("tab", { name: "Single File OCR" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: /Upload & Process/i })).toBeInTheDocument();
    expect(screen.queryByTestId("batch-file-input")).toBeNull();
  });

  it("lists every selected file with its size and rejects unsupported types", () => {
    renderPage();
    openMultiple();
    pick([pdf("a.pdf"), pdf("b.pdf"), new File(["x"], "virus.exe")]);

    expect(screen.getByText("2 files selected")).toBeInTheDocument();
    expect(screen.getByText("a.pdf")).toBeInTheDocument();
    expect(screen.getByText("b.pdf")).toBeInTheDocument();
    expect(screen.getByText(/Unsupported file type: virus\.exe/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Process 2 files" })).not.toBeDisabled();
  });

  it("rejects a file over the server's per-file size limit before ever uploading it", async () => {
    const { client } = renderPage();
    openMultiple();
    // Wait for the mocked GET /batches/capacity (max_file_mb: 1024) to
    // actually resolve and land in the query cache -- merely having been
    // *called* isn't enough: the size check below needs the resolved
    // limit, and the promise/react-query/re-render chain takes a few
    // microtask ticks after that call to actually settle.
    await waitFor(() => expect(client.getQueryData(["batchCapacity"])).toBeTruthy());

    const oversized = pdf("huge_scan.pdf");
    Object.defineProperty(oversized, "size", { value: 1100 * 1024 * 1024, configurable: true }); // 1100MB > the 1024MB mocked limit
    pick([pdf("a.pdf"), oversized]);

    expect(screen.getByText("1 file selected")).toBeInTheDocument();
    expect(screen.getByText("a.pdf")).toBeInTheDocument();
    expect(screen.queryByText("huge_scan.pdf")).toBeNull();
    expect(screen.getByText(/Too large \(max 1024MB\): huge_scan\.pdf/)).toBeInTheDocument();
    expect(batchApi.addFile).not.toHaveBeenCalled();
  });

  it("removes a file from the selection", () => {
    renderPage();
    openMultiple();
    pick([pdf("a.pdf"), pdf("b.pdf")]);
    fireEvent.click(screen.getByRole("button", { name: "Remove a.pdf" }));
    expect(screen.queryByText("a.pdf")).toBeNull();
    expect(screen.getByRole("button", { name: "Process 1 file" })).toBeInTheDocument();
  });

  it("disables the button until at least one file is chosen", () => {
    renderPage();
    openMultiple();
    expect(screen.getByRole("button", { name: "Process files" })).toBeDisabled();
  });

  it("applies the engine/DPI/profile chosen above to the whole batch, uploads in order, then opens the queue", async () => {
    (batchApi.addFile as any)
      .mockResolvedValueOnce({ job_id: "j1", total_pages: 2 })
      .mockResolvedValueOnce({ job_id: "j2", total_pages: 80 });
    renderPage();
    openMultiple();
    fireEvent.click(screen.getByRole("button", { name: /Tesseract/ }));
    fireEvent.change(screen.getAllByRole("combobox")[0], { target: { value: "300" } });
    pick([pdf("small.pdf"), pdf("big.pdf")]);
    fireEvent.click(screen.getByRole("button", { name: "Process 2 files" }));

    await waitFor(() => expect(mockedNavigate).toHaveBeenCalledWith("/batches/b1", { state: { skipped: [] } }));
    expect(batchApi.create).toHaveBeenCalledWith({ ocrProvider: "tesseract", dpi: 300, preprocessProfile: "FAST" });
    const uploaded = (batchApi.addFile as any).mock.calls.map((c: any[]) => [c[0], c[1].name]);
    expect(uploaded).toEqual([["b1", "small.pdf"], ["b1", "big.pdf"]]);
    expect(batchApi.start).toHaveBeenCalledWith("b1");
    expect((batchApi.create as any).mock.invocationCallOrder[0]).toBeLessThan((batchApi.start as any).mock.invocationCallOrder[0]);
  });

  it("one refused file does not stop the batch: the others are processed and the refusal is reported", async () => {
    (batchApi.addFile as any)
      .mockResolvedValueOnce({ job_id: "j1", total_pages: 2 })
      .mockRejectedValueOnce({ response: { data: { detail: "Corrupted or unreadable PDF" } } })
      .mockResolvedValueOnce({ job_id: "j3", total_pages: 5 });
    renderPage();
    openMultiple();
    pick([pdf("ok1.pdf"), pdf("bad.pdf"), pdf("ok2.pdf")]);
    fireEvent.click(screen.getByRole("button", { name: "Process 3 files" }));

    await waitFor(() => expect(batchApi.start).toHaveBeenCalledTimes(1));
    expect(batchApi.addFile).toHaveBeenCalledTimes(3);
    expect(mockedNavigate).toHaveBeenCalledWith("/batches/b1", {
      state: { skipped: [{ name: "bad.pdf", reason: "Corrupted or unreadable PDF" }] },
    });
  });

  it("does not start (and stays on the page) when every file is refused", async () => {
    (batchApi.addFile as any).mockRejectedValue({ response: { data: { detail: "Unsupported" } } });
    renderPage();
    openMultiple();
    pick([pdf("a.pdf")]);
    fireEvent.click(screen.getByRole("button", { name: "Process 1 file" }));

    expect(await screen.findByText(/None of the files could be added/)).toBeInTheDocument();
    expect(batchApi.start).not.toHaveBeenCalled();
    expect(mockedNavigate).not.toHaveBeenCalled();
    expect(screen.getByText("Unsupported")).toBeInTheDocument(); // per-file reason stays visible
  });

  it("shows the page count returned by the server next to an uploaded file", async () => {
    let release!: (v: unknown) => void;
    (batchApi.addFile as any).mockImplementationOnce(() => new Promise((r) => (release = r)));
    renderPage();
    openMultiple();
    pick([pdf("a.pdf")]);
    fireEvent.click(screen.getByRole("button", { name: "Process 1 file" }));

    const row = (await screen.findByText("a.pdf")).closest("li")!;
    expect(within(row).getByText(/Uploading/)).toBeInTheDocument();
    release({ job_id: "j1", total_pages: 12 });
    await waitFor(() => expect(within(row).getByText(/12 pages/)).toBeInTheDocument());
  });

  it("leaves the single-file flow intact after switching modes back and forth", () => {
    renderPage();
    const single = document.getElementById("file-input") as HTMLInputElement;
    fireEvent.change(single, { target: { files: [pdf("scan.pdf")] } });
    openMultiple();
    expect(screen.queryByRole("button", { name: /Upload & Process/i })).toBeNull(); // hidden while in multiple mode
    fireEvent.click(screen.getByRole("tab", { name: "Single File OCR" }));
    expect(screen.getByText("scan.pdf")).toBeInTheDocument(); // the earlier pick survived
    expect(screen.getByRole("button", { name: /Upload & Process/i })).not.toBeDisabled();
  });
});
