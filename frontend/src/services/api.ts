import axios from "axios";
import type {
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

  reconstruct: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/reconstruct`);
    return data;
  },

  exportPdf: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/export/pdf`);
    return data;
  },

  exportDocx: async (documentId: string) => {
    const { data } = await api.post(`/documents/${documentId}/export/docx`);
    return data;
  },

  downloadPdfUrl: (documentId: string) => `${API_BASE_URL}/documents/${documentId}/download/pdf`,
  downloadDocxUrl: (documentId: string) => `${API_BASE_URL}/documents/${documentId}/download/docx`,

  quality: async (documentId: string) => {
    const { data } = await api.get<QualityReport>(`/documents/${documentId}/quality`);
    return data;
  },
};
