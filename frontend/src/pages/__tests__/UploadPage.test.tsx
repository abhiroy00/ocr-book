import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import UploadPage from "@/pages/UploadPage";
import { documentsApi } from "@/services/api";

vi.mock("@/services/api", () => ({
  documentsApi: { upload: vi.fn() },
  fileUrl: (p: string) => p,
}));

const mockedNavigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => mockedNavigate };
});

function renderPage() {
  return render(
    <MemoryRouter>
      <UploadPage />
    </MemoryRouter>,
  );
}

describe("UploadPage", () => {
  it("rejects an unsupported file extension", async () => {
    renderPage();
    const input = screen.getByLabelText(/drag/i, { selector: "input" }) as HTMLInputElement;
    const file = new File(["x"], "malware.exe", { type: "application/octet-stream" });
    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByText(/Unsupported file type/i)).toBeInTheDocument();
  });

  it("accepts a valid PDF and enables the upload button", async () => {
    renderPage();
    const input = document.getElementById("file-input") as HTMLInputElement;
    const file = new File(["%PDF-1.4"], "scan.pdf", { type: "application/pdf" });
    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByText("scan.pdf")).toBeInTheDocument();
    const uploadButton = screen.getByRole("button", { name: /Upload & Process/i });
    expect(uploadButton).not.toBeDisabled();
  });

  it("navigates to the document detail page after a successful upload", async () => {
    (documentsApi.upload as any).mockResolvedValue({
      document_id: "doc-123",
      job_id: "job-1",
      status: "QUEUED",
      original_filename: "scan.pdf",
      page_count: 1,
    });

    renderPage();
    const input = document.getElementById("file-input") as HTMLInputElement;
    const file = new File(["%PDF-1.4"], "scan.pdf", { type: "application/pdf" });
    fireEvent.change(input, { target: { files: [file] } });

    const uploadButton = await screen.findByRole("button", { name: /Upload & Process/i });
    fireEvent.click(uploadButton);

    await waitFor(() => expect(mockedNavigate).toHaveBeenCalledWith("/documents/doc-123"));
  });
});
