export type DocumentStatus =
  | "UPLOADED"
  | "QUEUED"
  | "PROCESSING"
  | "OCR_PROCESSING"
  | "LAYOUT_PROCESSING"
  | "TABLE_PROCESSING"
  | "RECONSTRUCTING"
  | "EXPORTING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export type ProcessingStage =
  | "upload"
  | "render"
  | "preprocess"
  | "ocr"
  | "layout"
  | "table"
  | "document_json"
  | "reconstruct"
  | "export"
  | "quality"
  | "done";

export type OCRProvider = "paddleocr" | "tesseract" | "nvidia" | "ollama";
export type PreprocessProfile = "FAST" | "BALANCED" | "HIGH_QUALITY";

export type LayoutBlockType =
  | "title"
  | "heading"
  | "subheading"
  | "paragraph"
  | "table"
  | "image"
  | "chart"
  | "signature"
  | "stamp"
  | "handwritten"
  | "header"
  | "footer"
  | "page_number"
  | "footnote"
  | "horizontal_line"
  | "vertical_line";

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DocumentRecord {
  id: string;
  original_filename: string;
  file_extension: string;
  mime_type: string;
  file_size_bytes: number;
  page_count: number;
  status: DocumentStatus;
  ocr_provider: OCRProvider;
  dpi: number;
  preprocess_profile: PreprocessProfile;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: DocumentRecord[];
  total: number;
  page: number;
  page_size: number;
}

export interface DocumentStats {
  total: number;
  processing: number;
  completed: number;
  failed: number;
}

export interface ProcessingJob {
  id: string;
  document_id: string;
  status: DocumentStatus;
  stage: ProcessingStage;
  progress_percent: number;
  ocr_provider: OCRProvider;
  dpi: number;
  preprocess_profile: PreprocessProfile;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  // Authoritative total pipeline duration, computed backend-side --
  // null while the job hasn't started or is still running (the frontend
  // Process Timer ticks live off `started_at` for that case), populated
  // once `finished_at` is set (COMPLETED/FAILED/CANCELLED all set this
  // together -- see `document_service.update_job_progress`).
  processing_duration_seconds: number | null;
}

export interface ProgressEvent {
  document_id: string;
  job_id: string;
  stage: ProcessingStage;
  percent: number;
  page: number | null;
  message: string;
  status: DocumentStatus;
}

export interface PageRecord {
  id: string;
  page_number: number;
  width: number;
  height: number;
  dpi: number;
  rotation: number;
  original_image_url: string;
  processed_image_url: string | null;
  ocr_status: string;
  layout_status: string;
  table_status: string;
  ocr_confidence_avg: number | null;
  layout_confidence_avg: number | null;
  table_confidence_avg: number | null;
  visual_similarity_score: number | null;
}

export interface OCRBlockRecord {
  id: string;
  page_id: string;
  block_id: string;
  line_id: string;
  text: string;
  original_text: string;
  confidence: number;
  bbox: BBox;
  polygon: number[][];
  language: string;
  is_reviewed: boolean;
}

export interface LayoutBlockRecord {
  id: string;
  page_id: string;
  block_type: LayoutBlockType;
  bbox: BBox;
  confidence: number;
  z_order: number;
  content: { text?: string; word_ids?: string[] };
  style: Record<string, unknown>;
  is_edited: boolean;
  edited_text: string | null;
}

export interface TableCellRecord {
  id: string;
  table_id: string;
  row: number;
  column: number;
  rowspan: number;
  colspan: number;
  text: string;
  bbox: BBox;
  align_h: "LEFT" | "CENTER" | "RIGHT";
  align_v: "TOP" | "MIDDLE" | "BOTTOM";
  confidence: number;
  is_header: boolean;
  is_edited: boolean;
}

export interface TableRecord {
  id: string;
  page_id: string;
  bbox: BBox;
  n_rows: number;
  n_cols: number;
  confidence: number;
  detection_method: string;
  column_widths: number[];
  row_heights: number[];
  border_style: Record<string, unknown>;
  cells: TableCellRecord[];
}

export interface QualityPageReport {
  page_number: number;
  ocr_confidence_avg: number | null;
  layout_confidence_avg: number | null;
  table_confidence_avg: number | null;
  visual_similarity_score: number | null;
  needs_review: boolean;
}

export interface QualityReport {
  document_id: string;
  pages: QualityPageReport[];
  overall_visual_similarity: number | null;
}

// Library accession register (cumulative Master Excel) -- see
// backend/app/services/accession_service.py.
export interface AccessionSummary {
  total_records: number;
  needs_review_count: number;
  latest_record_date: string | null;
  total_documents_processed: number;
}
