import axios from "axios";
import type {
  AccessionSummary,
  DocumentListResponse,
  DocumentRecord,
  DocumentStats,
  LayoutBlockRecord,
  OCRBlockRecord,
  PageRecord,
  ProcessingJob,
  QualityReport,
  TableRecord,
  OCRProvider,
  PreprocessProfile,
} from "@/types/document";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

export const api = axios.create({ baseURL: API_BASE_URL });

export function wsUrlForDocument(documentId: string): string {
  const base = import.meta.env.VITE_WS_BASE_URL || "ws://localhost:8000/ws";
  return `${base}/documents/${documentId}`;
}

export function fileUrl(path: string): string {
  if (path.startsWith("http")) return path;
  const httpBase = API_BASE_URL.replace(/\/api\/?$/, "");
  return path.startsWith("/api") ? `${httpBase}${path}` : `${httpBase}${path.startsWith("/") ? "" : "/"}${path}`;
}

export interface UploadOptions {
  ocrProvider?: OCRProvider;
  dpi?: number;
  preprocessProfile?: PreprocessProfile;
  onProgress?: (percent: number) => void;
}

export const documentsApi = {
  upload: async (file: File, options: UploadOptions = {}) => {
    const formData = new FormData();
    formData.append("file", file);
    if (options.ocrProvider) formData.append("ocr_provider", options.ocrProvider);
    if (options.dpi) formData.append("dpi", String(options.dpi));
    if (options.preprocessProfile) formData.append("preprocess_profile", options.preprocessProfile);

    const { data } = await api.post("/documents/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (evt) => {
        if (options.onProgress && evt.total) {
          options.onProgress(Math.round((evt.loaded / evt.total) * 100));
        }
      },
    });
    return data as { document_id: string; job_id: string; status: string; original_filename: string; page_count: number };
  },

  list: async (params: { status?: string; page?: number; page_size?: number } = {}) => {
    const { data } = await api.get<DocumentListResponse>("/documents", { params });
    return data;
  },

  stats: async () => {
    const { data } = await api.get<DocumentStats>("/documents/stats/summary");
    return data;
  },

  get: async (documentId: string) => {
    const { data } = await api.get<DocumentRecord>(`/documents/${documentId}`);
    return data;
  },

  remove: async (documentId: string) => {
    await api.delete(`/documents/${documentId}`);
  },

  reprocess: async (documentId: string, options: UploadOptions = {}) => {
    const { data } = await api.post<ProcessingJob>(`/documents/${documentId}/process`, {
      ocr_provider: options.ocrProvider,
      dpi: options.dpi,
      preprocess_profile: options.preprocessProfile,
    });
    return data;
  },

  progress: async (documentId: string) => {
    const { data } = await api.get<ProcessingJob>(`/documents/${documentId}/progress`);
    return data;
  },

  cancel: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/cancel`);
    return data as { status: string };
  },

  pages: async (documentId: string) => {
    const { data } = await api.get<PageRecord[]>(`/documents/${documentId}/pages`);
    return data;
  },

  page: async (documentId: string, pageNumber: number) => {
    const { data } = await api.get<PageRecord>(`/documents/${documentId}/pages/${pageNumber}`);
    return data;
  },

  pageJson: async (documentId: string, pageNumber: number) => {
    const { data } = await api.get(`/documents/${documentId}/pages/${pageNumber}/json`);
    return data;
  },

  ocrBlocks: async (documentId: string, page?: number) => {
    const { data } = await api.get<OCRBlockRecord[]>(`/documents/${documentId}/ocr`, { params: { page } });
    return data;
  },

  layoutBlocks: async (documentId: string, page?: number) => {
    const { data } = await api.get<LayoutBlockRecord[]>(`/documents/${documentId}/layout`, { params: { page } });
    return data;
  },

  tables: async (documentId: string, page?: number) => {
    const { data } = await api.get<TableRecord[]>(`/documents/${documentId}/tables`, { params: { page } });
    return data;
  },

  updateBlock: async (documentId: string, blockId: string, body: { text?: string; style?: Record<string, unknown> }) => {
    const { data } = await api.put<LayoutBlockRecord>(`/documents/${documentId}/blocks/${blockId}`, body);
    return data;
  },

  updateCell: async (
    documentId: string,
    cellId: string,
    body: { text?: string; align_h?: string; align_v?: string; rowspan?: number; colspan?: number },
  ) => {
    const { data } = await api.put(`/documents/${documentId}/cells/${cellId}`, body);
    return data;
  },

  updateTableStructure: async (
    documentId: string,
    tableId: string,
    body: { op: string; row?: number; column?: number; cell_ids?: string[] },
  ) => {
    const { data } = await api.put<TableRecord>(`/documents/${documentId}/tables/${tableId}`, body);
    return data;
  },

  // Every regenerate/export action below is queued as an async Celery
  // task server-side (see backend `_dispatch_export`) rather than run
  // inline in the HTTP request -- a large (100+ page) document's
  // regenerate could otherwise take longer than any reasonable client
  // timeout while still completing correctly server-side, making the
  // action look broken even though nothing failed. `pollExportStatus`
  // dispatches, then polls status until the task finishes, so from a
  // caller's point of view (e.g. a `useMutation`) the returned promise
  // still only resolves once the file is actually ready -- no other
  // UI code needs to change to get the non-blocking behavior.
  pollExportStatus: async (documentId: string, taskId: string, { intervalMs = 2000, timeoutMs = 30 * 60 * 1000 } = {}) => {
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
      const { data } = await api.get(`/documents/${documentId}/export/status/${taskId}`);
      if (data.state === "SUCCESS") return data.result;
      if (data.state === "FAILURE") throw new Error(data.error || "Export failed");
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
    throw new Error("Export timed out waiting for the background task to finish");
  },

  reconstruct: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/reconstruct`);
    return documentsApi.pollExportStatus(documentId, data.task_id);
  },

  exportSearchablePdf: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/export/pdf/searchable`);
    return documentsApi.pollExportStatus(documentId, data.task_id);
  },

  exportPdf: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/export/pdf`);
    return documentsApi.pollExportStatus(documentId, data.task_id);
  },

  exportDocx: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/export/docx`);
    return documentsApi.pollExportStatus(documentId, data.task_id);
  },

  exportExcel: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/export/excel`);
    return documentsApi.pollExportStatus(documentId, data.task_id);
  },

  downloadPdfUrl: (documentId: string) => `${API_BASE_URL}/documents/${documentId}/download/pdf`,
  downloadReconstructedPdfUrl: (documentId: string) => `${API_BASE_URL}/documents/${documentId}/download/pdf/reconstructed`,
  downloadDocxUrl: (documentId: string) => `${API_BASE_URL}/documents/${documentId}/download/docx`,
  downloadExcelUrl: (documentId: string) => `${API_BASE_URL}/documents/${documentId}/download/excel`,

  quality: async (documentId: string) => {
    const { data } = await api.get<QualityReport>(`/documents/${documentId}/quality`);
    return data;
  },
};

// Library accession register (cumulative Master Excel) -- see
// backend/app/api/v1/accession.py. Kept as its own small object rather
// than folded into `documentsApi` since it's a distinct resource (one row
// per processed document's extracted bibliographic metadata, not a
// per-document action).
export const accessionApi = {
  summary: async () => {
    const { data } = await api.get<AccessionSummary>("/accession-records/summary");
    return data;
  },

  masterExcelUrl: () => `${API_BASE_URL}/accession-records/export/master-excel`,
};
