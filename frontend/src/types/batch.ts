import type { DocumentStatus, OCRProvider, PreprocessProfile, ProcessingStage } from "@/types/document";

// pending -> queued -> running -> completed | failed | cancelled ("duplicate"
// = identical content already existed, so it was not reprocessed).
export type BatchItemState = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled" | "duplicate";

export type BatchStatus = "uploading" | "processing" | "completed" | "completed_with_errors" | "cancelled";

export interface BatchItem {
  job_id: string;
  document_id: string;
  processing_job_id: string | null;
  filename: string;
  file_size_bytes: number;
  position: number;
  state: BatchItemState;
  status: DocumentStatus | null;
  stage: ProcessingStage | null;
  progress_percent: number;
  processed_pages: number;
  total_pages: number;
  priority_score: number;
  attempts: number;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
}

export interface ConcurrencyInfo {
  provider: string;
  limit: number;
  limiting_factor: string;
  cpu_count: number;
  total_mem_mb: number | null;
  bounds: Record<string, number>;
}

export interface Batch {
  batch_id: string;
  status: BatchStatus;
  ocr_provider: OCRProvider;
  dpi: number;
  preprocess_profile: PreprocessProfile;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_seconds: number | null;
  total_files: number;
  completed_files: number;
  failed_files: number;
  cancelled_files: number;
  duplicate_files: number;
  queued_files: number;
  processing_files: number;
  finished_files: number;
  total_pages: number;
  processed_pages: number;
  overall_percent: number;
  concurrency: ConcurrencyInfo | null;
  items: BatchItem[];
}

export interface BatchCapacity {
  concurrency: ConcurrencyInfo;
  max_files: number;
  max_total_mb: number;
  max_file_mb: number;
  allowed_extensions: string[];
}

/** A file the server refused while a batch was being uploaded (bad type,
 * corrupt, over a limit, ...). The rest of the batch is still processed. */
export interface SkippedFile {
  name: string;
  reason: string;
}
