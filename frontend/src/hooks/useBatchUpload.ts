import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { batchApi } from "@/services/batchApi";
import type { OCRProvider, PreprocessProfile } from "@/types/document";
import type { SkippedFile } from "@/types/batch";

export type FileUploadState = "selected" | "uploading" | "uploaded" | "rejected";

export interface SelectedFile {
  id: string;
  file: File;
  state: FileUploadState;
  percent: number;
  pages: number | null; // known only once the server has read the file
  error: string | null;
}

interface Options {
  accepted: string[];
  ocrProvider: OCRProvider;
  dpi: number;
  profile: PreprocessProfile;
}

let nextId = 0;
const newId = () => `f${(nextId += 1)}`;

function errorText(err: any, fallback: string): string {
  const detail = err?.response?.data?.detail;
  return typeof detail === "string" ? detail : fallback;
}

/**
 * Client side of Multiple PDF OCR: holds the selected files and runs the
 * create -> upload each -> start sequence, then opens the queue page.
 *
 * Files are uploaded one at a time (bounded memory/bandwidth, and each
 * request stays within the normal single-file size). A file the server
 * refuses does not abort the rest -- it is reported and skipped.
 */
export function useBatchUpload({ accepted, ocrProvider, dpi, profile }: Options) {
  const navigate = useNavigate();
  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const addFiles = useCallback(
    (incoming: FileList | File[] | null) => {
      if (!incoming || incoming.length === 0) return;
      const rejected: string[] = [];
      const ok: SelectedFile[] = [];
      for (const file of Array.from(incoming)) {
        const ext = "." + file.name.split(".").pop()?.toLowerCase();
        if (!accepted.includes(ext)) {
          rejected.push(file.name);
          continue;
        }
        ok.push({ id: newId(), file, state: "selected", percent: 0, pages: null, error: null });
      }
      setError(rejected.length ? `Unsupported file type: ${rejected.join(", ")}. Allowed: ${accepted.join(", ")}` : null);
      if (ok.length) setFiles((prev) => [...prev, ...ok]);
    },
    [accepted],
  );

  const removeFile = useCallback((id: string) => setFiles((prev) => prev.filter((f) => f.id !== id)), []);
  const clear = useCallback(() => {
    setFiles([]);
    setError(null);
  }, []);

  const patch = useCallback(
    (id: string, update: Partial<SelectedFile>) =>
      setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, ...update } : f))),
    [],
  );

  const submit = useCallback(async () => {
    if (files.length === 0 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const batch = await batchApi.create({ ocrProvider, dpi, preprocessProfile: profile });
      const skipped: SkippedFile[] = [];
      let added = 0;
      for (const entry of files) {
        patch(entry.id, { state: "uploading", percent: 0, error: null });
        try {
          const item = await batchApi.addFile(batch.batch_id, entry.file, (percent) => patch(entry.id, { percent }));
          patch(entry.id, { state: "uploaded", percent: 100, pages: item.total_pages });
          added += 1;
        } catch (err: any) {
          const reason = errorText(err, "Upload failed");
          patch(entry.id, { state: "rejected", error: reason });
          skipped.push({ name: entry.file.name, reason });
        }
      }
      if (added === 0) {
        setError("None of the files could be added. Fix the problems listed below and try again.");
        return;
      }
      await batchApi.start(batch.batch_id);
      navigate(`/batches/${batch.batch_id}`, { state: { skipped } });
    } catch (err: any) {
      setError(errorText(err, "Could not start the batch. Please try again."));
    } finally {
      setBusy(false);
    }
  }, [files, busy, ocrProvider, dpi, profile, navigate, patch]);

  return { files, error, busy, addFiles, removeFile, clear, submit };
}
