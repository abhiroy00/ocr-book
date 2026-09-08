import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { documentsApi } from "@/services/api";
import type { OCRProvider, PreprocessProfile } from "@/types/document";

const ACCEPTED_EXTENSIONS = [".pdf", ".jpg", ".jpeg", ".png", ".webp"];
const DPI_OPTIONS = [150, 200, 300, 400, 600];
const OCR_OPTIONS: { value: OCRProvider; label: string; hint: string }[] = [
  { value: "paddleocr", label: "PaddleOCR", hint: "Default — local, no API key required" },
  { value: "tesseract", label: "Tesseract", hint: "Offline fallback engine" },
  { value: "nvidia", label: "NVIDIA VLM", hint: "Optional — requires NVIDIA_API_KEY" },
  { value: "ollama", label: "Ollama", hint: "Optional — requires a local Ollama server" },
];
const PROFILE_OPTIONS: { value: PreprocessProfile; label: string; hint: string }[] = [
  { value: "FAST", label: "Fast", hint: "Deskew + grayscale + mild denoise" },
  { value: "BALANCED", label: "Balanced", hint: "+ background removal + adaptive threshold" },
  { value: "HIGH_QUALITY", label: "High Quality", hint: "+ dewarp + advanced denoise + contrast enhancement" },
];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(1)} ${units[i]}`;
}

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [ocrProvider, setOcrProvider] = useState<OCRProvider>("paddleocr");
  // FAST/150 is the safer default for CPU-only PaddleOCR on typical
  // hardware — BALANCED/300 can turn a large multi-page document into an
  // hours-long job. Users who want maximum quality on a small document can
  // still pick BALANCED/HIGH_QUALITY and a higher DPI from the dropdowns.
  const [dpi, setDpi] = useState(150);
  const [profile, setProfile] = useState<PreprocessProfile>("FAST");
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;
    const picked = files[0];
    const ext = "." + picked.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported file type ${ext}. Allowed: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return;
    }
    setError(null);
    setFile(picked);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const startUpload = async () => {
    if (!file) return;
    setUploadPercent(0);
    setError(null);
    try {
      const result = await documentsApi.upload(file, {
        ocrProvider,
        dpi,
        preprocessProfile: profile,
        onProgress: setUploadPercent,
      });
      navigate(`/documents/${result.document_id}`);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Upload failed. Please try again.");
      setUploadPercent(null);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-slate-900">Upload a scanned document</h1>
        <p className="text-slate-500 mt-1">PDF, JPG, PNG or WebP. Layout, tables and images are preserved during reconstruction.</p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={clsx(
          "border-2 border-dashed rounded-2xl p-12 text-center transition-colors bg-white",
          dragOver ? "border-brand-500 bg-brand-50" : "border-slate-300",
        )}
      >
        <input
          id="file-input"
          type="file"
          className="hidden"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          onChange={(e) => handleFiles(e.target.files)}
        />
        {!file ? (
          <label htmlFor="file-input" className="cursor-pointer">
            <div className="text-4xl mb-3">📄</div>
            <div className="font-medium text-slate-900">Drag & drop a file here, or click to browse</div>
            <div className="text-sm text-slate-500 mt-1">{ACCEPTED_EXTENSIONS.join(", ")}</div>
          </label>
        ) : (
          <div>
            <div className="text-4xl mb-3">✅</div>
            <div className="font-medium text-slate-900">{file.name}</div>
            <div className="text-sm text-slate-500 mt-1">{formatBytes(file.size)}</div>
            <button className="mt-3 text-sm text-brand-600 hover:underline" onClick={() => setFile(null)}>
              Choose a different file
            </button>
          </div>
        )}
      </div>

      {error && <div className="rounded-lg bg-red-50 text-red-700 text-sm px-4 py-3">{error}</div>}

      <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-5">
        <div>
          <label className="text-sm font-medium text-slate-700">OCR Engine</label>
          <div className="grid grid-cols-2 gap-2 mt-2">
            {OCR_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setOcrProvider(opt.value)}
                className={clsx(
                  "text-left px-3 py-2 rounded-lg border text-sm",
                  ocrProvider === opt.value ? "border-brand-500 bg-brand-50" : "border-slate-200 hover:bg-slate-50",
                )}
              >
                <div className="font-medium text-slate-900">{opt.label}</div>
                <div className="text-xs text-slate-500">{opt.hint}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-5">
          <div>
            <label className="text-sm font-medium text-slate-700">Render DPI</label>
            <select
              value={dpi}
              onChange={(e) => setDpi(Number(e.target.value))}
              className="mt-2 w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
            >
              {DPI_OPTIONS.map((d) => (
                <option key={d} value={d}>
                  {d} DPI
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium text-slate-700">Preprocessing profile</label>
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value as PreprocessProfile)}
              className="mt-2 w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
            >
              {PROFILE_OPTIONS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-slate-400 mt-1">{PROFILE_OPTIONS.find((p) => p.value === profile)?.hint}</p>
          </div>
        </div>
      </div>

      {uploadPercent !== null && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex justify-between text-sm mb-1">
            <span>Uploading…</span>
            <span>{uploadPercent}%</span>
          </div>
          <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
            <div className="h-full bg-brand-600 transition-all" style={{ width: `${uploadPercent}%` }} />
          </div>
        </div>
      )}

      <button
        disabled={!file || uploadPercent !== null}
        onClick={startUpload}
        className="w-full py-3 rounded-lg bg-brand-600 text-white font-medium disabled:opacity-40 hover:bg-brand-700"
      >
        {uploadPercent !== null ? "Uploading…" : "Upload & Process"}
      </button>
    </div>
  );
}
