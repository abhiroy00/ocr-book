import { api } from "@/services/api";
import type { OCRProvider, PreprocessProfile } from "@/types/document";
import type { Batch, BatchCapacity, BatchItem } from "@/types/batch";

// Multiple-file OCR endpoints (backend/app/api/v1/batches.py). Kept in its
// own module, separate from `documentsApi`, so the single-file client is
// untouched; it shares the same axios instance (base URL, interceptors).
export interface CreateBatchOptions {
  ocrProvider?: OCRProvider;
  dpi?: number;
  preprocessProfile?: PreprocessProfile;
}

export const batchApi = {
  capacity: async (ocrProvider?: OCRProvider) => {
    const { data } = await api.get<BatchCapacity>("/batches/capacity", { params: { ocr_provider: ocrProvider } });
    return data;
  },

  create: async (options: CreateBatchOptions = {}) => {
    const { data } = await api.post<Batch>("/batches", {
      ocr_provider: options.ocrProvider,
      dpi: options.dpi,
      preprocess_profile: options.preprocessProfile,
    });
    return data;
  },

  // One file per request: each stays within the normal single-file upload
  // size, and only one file is in flight at a time.
  addFile: async (batchId: string, file: File, onProgress?: (percent: number) => void) => {
    const formData = new FormData();
    formData.append("file", file);
    const { data } = await api.post<BatchItem>(`/batches/${batchId}/files`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (evt) => {
        if (onProgress && evt.total) onProgress(Math.round((evt.loaded / evt.total) * 100));
      },
    });
    return data;
  },

  start: async (batchId: string) => {
    const { data } = await api.post<{ batch_id: string; queued_files: number; status: string }>(`/batches/${batchId}/start`);
    return data;
  },

  status: async (batchId: string) => {
    const { data } = await api.get<Batch>(`/batches/${batchId}`);
    return data;
  },

  retry: async (batchId: string, jobId: string) => {
    const { data } = await api.post<BatchItem>(`/batches/${batchId}/items/${jobId}/retry`);
    return data;
  },

  cancel: async (batchId: string) => {
    const { data } = await api.post<{ status: string; cancelled_waiting: number; cancelling_running: number }>(
      `/batches/${batchId}/cancel`,
    );
    return data;
  },
};
