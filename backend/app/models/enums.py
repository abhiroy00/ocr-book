from __future__ import annotations

import enum


class DocumentStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    OCR_PROCESSING = "OCR_PROCESSING"
    LAYOUT_PROCESSING = "LAYOUT_PROCESSING"
    TABLE_PROCESSING = "TABLE_PROCESSING"
    RECONSTRUCTING = "RECONSTRUCTING"
    EXPORTING = "EXPORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProcessingStage(str, enum.Enum):
    UPLOAD = "upload"
    RENDER = "render"
    PREPROCESS = "preprocess"
    OCR = "ocr"
    LAYOUT = "layout"
    TABLE = "table"
    DOCUMENT_JSON = "document_json"
    RECONSTRUCT = "reconstruct"
    EXPORT = "export"
    QUALITY = "quality"
    DONE = "done"


class LayoutBlockType(str, enum.Enum):
    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    SIGNATURE = "signature"
    STAMP = "stamp"
    HANDWRITTEN = "handwritten"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FOOTNOTE = "footnote"
    HLINE = "horizontal_line"
    VLINE = "vertical_line"


class TextAlign(str, enum.Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class VerticalAlign(str, enum.Enum):
    TOP = "TOP"
    MIDDLE = "MIDDLE"
    BOTTOM = "BOTTOM"


class TableDetectionMethod(str, enum.Enum):
    OPENCV_LINES = "opencv_lines"
    PADDLE_STRUCTURE = "paddle_structure"
    OCR_FALLBACK = "ocr_fallback"


class ExportType(str, enum.Enum):
    CLEAN_PDF = "clean_pdf"
    SEARCHABLE_PDF = "searchable_pdf"
    RECONSTRUCTED_PDF = "reconstructed_pdf"
    DOCX = "docx"
    XLSX = "xlsx"


class OCRProviderEnum(str, enum.Enum):
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"


class PreprocessProfileEnum(str, enum.Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    HIGH_QUALITY = "HIGH_QUALITY"
