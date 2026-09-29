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
  // Server's own per-file cap (from GET /batches/capacity, in MB) --
  // when known, a too-large file is rejected immediately with a clear
  // reason instead of only after an upload attempt nginx would reject
  // anyway (see `errorText`'s 413 case, which stays as the fallback for
  // whenever this isn't available yet or the two ever drift apart).
  maxFileMb?: number;
}

let nextId = 0;
const newId = () => `f${(nextId += 1)}`;

function errorText(err: any, fallback: string): string {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  // A server error normally arrives as FastAPI's JSON `{"detail": "..."}`.
  // When nginx itself rejects a request before it ever reaches the
  // backend -- most commonly a 413 for a file over `client_max_body_size`
  // -- the response body is nginx's own plain HTML error page instead, so
  // `detail` is never there to read. Confirmed as a real case: a 770MB
  // scan was silently reported as just "Upload failed" with no size
  // mentioned anywhere, giving no way to tell a size problem apart from
  // any other failure. Naming the status code at least narrows it down.
  const status = err?.response?.status;
  if (status === 413) return "File is too large for the server's upload limit.";
  return status ? `${fallback} (HTTP ${status})` : fallback;
}

/**
 * Client side of Multiple PDF OCR: holds the selected files and runs the
 * create -> upload each -> start sequence, then opens the queue page.
 *
 * Files are uploaded one at a time (bounded memory/bandwidth, and each
 * request stays within the normal single-file size). A file the server
 * refuses does not abort the rest -- it is reported and skipped.
 */
export function useBatchUpload({ accepted, ocrProvider, dpi, profile, maxFileMb }: Options) {
  const navigate = useNavigate();
  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const addFiles = useCallback(
    (incoming: FileList | File[] | null) => {
      if (!incoming || incoming.length === 0) return;
      const badType: string[] = [];
      const tooLarge: string[] = [];
      const ok: SelectedFile[] = [];
      const maxBytes = maxFileMb ? maxFileMb * 1024 * 1024 : null;
      for (const file of Array.from(incoming)) {
        const ext = "." + file.name.split(".").pop()?.toLowerCase();
        if (!accepted.includes(ext)) {
          badType.push(file.name);
          continue;
        }
        if (maxBytes && file.size > maxBytes) {
          tooLarge.push(`${file.name} (${(file.size / (1024 * 1024)).toFixed(0)}MB)`);
          continue;
        }
        ok.push({ id: newId(), file, state: "selected", percent: 0, pages: null, error: null });
      }
      const messages: string[] = [];
      if (badType.length) messages.push(`Unsupported file type: ${badType.join(", ")}. Allowed: ${accepted.join(", ")}`);
      if (tooLarge.length) messages.push(`Too large (max ${maxFileMb}MB): ${tooLarge.join(", ")}`);
      setError(messages.length ? messages.join(" ") : null);
      if (ok.length) setFiles((prev) => [...prev, ...ok]);
    },
    [accepted, maxFileMb],
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
