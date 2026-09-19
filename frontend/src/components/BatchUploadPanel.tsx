import { useState } from "react";
import clsx from "clsx";
import type { SelectedFile } from "@/hooks/useBatchUpload";

export function formatBytes(bytes: number): string {
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

const STATE_LABEL: Record<SelectedFile["state"], string> = {
  selected: "Ready",
  uploading: "Uploading…",
  uploaded: "Uploaded",
  rejected: "Rejected",
};

/** Multi-file drop zone + the list of selected files. Same look as the
 * single-file drop zone; it only differs in accepting many files. */
export function BatchDropzone({
  files,
  accepted,
  busy,
  onAdd,
  onRemove,
  onClear,
}: {
  files: SelectedFile[];
  accepted: string[];
  busy: boolean;
  onAdd: (files: FileList | null) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}) {
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (!busy) onAdd(e.dataTransfer.files);
      }}
      className={clsx(
        "border-2 border-dashed rounded-2xl p-8 bg-white transition-colors",
        dragOver ? "border-brand-500 bg-brand-50" : "border-slate-300",
      )}
    >
      <input
        id="batch-file-input"
        data-testid="batch-file-input"
        type="file"
        multiple
        className="hidden"
        accept={accepted.join(",")}
        disabled={busy}
        onChange={(e) => {
          onAdd(e.target.files);
          e.target.value = ""; // allow re-selecting the same file after removing it
        }}
      />
      <label htmlFor="batch-file-input" className={clsx("block text-center", busy ? "" : "cursor-pointer")}>
        <div className="text-4xl mb-3">📚</div>
        <div className="font-medium text-slate-900">Drag & drop several files here, or click to browse</div>
        <div className="text-sm text-slate-500 mt-1">{accepted.join(", ")} — select as many as you need</div>
      </label>

      {files.length > 0 && (
        <div className="mt-6">
          <div className="flex justify-between items-center text-sm mb-2">
            <span className="font-medium text-slate-700">
              {files.length} file{files.length === 1 ? "" : "s"} selected
            </span>
            {!busy && (
              <button className="text-brand-600 hover:underline" onClick={onClear}>
                Clear all
              </button>
            )}
          </div>
          <ul className="divide-y divide-slate-100 border border-slate-200 rounded-lg text-sm">
            {files.map((f, i) => (
              <li key={f.id} className="px-3 py-2">
                <div className="flex items-center gap-3">
                  <span className="text-slate-400 w-5 text-right">{i + 1}</span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-medium text-slate-900">{f.file.name}</div>
                    <div className="text-xs text-slate-500">
                      {formatBytes(f.file.size)}
                      {f.pages !== null && ` · ${f.pages} page${f.pages === 1 ? "" : "s"}`}
                    </div>
                  </div>
                  <span
                    className={clsx(
                      "text-xs px-2 py-0.5 rounded-full",
                      f.state === "rejected" && "bg-red-100 text-red-800",
                      f.state === "uploaded" && "bg-emerald-100 text-emerald-800",
                      f.state === "uploading" && "bg-amber-100 text-amber-800",
                      f.state === "selected" && "bg-slate-100 text-slate-700",
                    )}
                  >
                    {f.state === "uploading" ? `${STATE_LABEL[f.state]} ${f.percent}%` : STATE_LABEL[f.state]}
                  </span>
                  {!busy && f.state !== "uploaded" && (
                    <button
                      aria-label={`Remove ${f.file.name}`}
                      className="text-slate-400 hover:text-red-600"
                      onClick={() => onRemove(f.id)}
                    >
                      ✕
                    </button>
                  )}
                </div>
                {f.error && <div className="text-xs text-red-700 mt-1 ml-8">{f.error}</div>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** Overall upload progress + the start button for Multiple PDF OCR. */
export function BatchSubmit({ files, busy, onSubmit }: { files: SelectedFile[]; busy: boolean; onSubmit: () => void }) {
  const done = files.filter((f) => f.state === "uploaded" || f.state === "rejected").length;
  const percent = files.length ? Math.round((files.reduce((sum, f) => sum + (f.state === "selected" ? 0 : f.percent), 0) / files.length)) : 0;

  return (
    <>
      {busy && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex justify-between text-sm mb-1">
            <span>
              Uploading {done} / {files.length} files…
            </span>
            <span>{percent}%</span>
          </div>
          <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
            <div className="h-full bg-brand-600 transition-all" style={{ width: `${percent}%` }} />
          </div>
        </div>
      )}
      <button
        disabled={files.length === 0 || busy}
        onClick={onSubmit}
        className="w-full py-3 rounded-lg bg-brand-600 text-white font-medium disabled:opacity-40 hover:bg-brand-700"
      >
        {busy ? "Uploading…" : files.length === 0 ? "Process files" : `Process ${files.length} file${files.length === 1 ? "" : "s"}`}
      </button>
    </>
  );
}
