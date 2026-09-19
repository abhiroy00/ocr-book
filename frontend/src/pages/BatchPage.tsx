import { Link, useLocation, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import ProgressBar from "@/components/ProgressBar";
import { formatBytes } from "@/components/BatchUploadPanel";
import { batchApi } from "@/services/batchApi";
import { documentsApi } from "@/services/api";
import type { Batch, BatchItem, BatchItemState, SkippedFile } from "@/types/batch";

const PILL: Record<BatchItemState, { label: string; className: string }> = {
  pending: { label: "Queued", className: "bg-slate-100 text-slate-700" },
  queued: { label: "Queued", className: "bg-slate-100 text-slate-700" },
  running: { label: "⏳ Processing", className: "bg-amber-100 text-amber-800" },
  completed: { label: "✓ Completed", className: "bg-emerald-100 text-emerald-800" },
  failed: { label: "✗ Failed", className: "bg-red-100 text-red-800" },
  cancelled: { label: "Cancelled", className: "bg-slate-200 text-slate-600" },
  duplicate: { label: "Already processed", className: "bg-sky-100 text-sky-800" },
};

function formatElapsed(totalSeconds: number | null): string {
  if (totalSeconds === null) return "";
  const s = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

function pageWord(n: number): string {
  return `${n} page${n === 1 ? "" : "s"}`;
}

function FileRow({
  index,
  item,
  onRetry,
  retrying,
}: {
  index: number;
  item: BatchItem;
  onRetry: (jobId: string) => void;
  retrying: boolean;
}) {
  const pill = PILL[item.state];
  const showProgress = item.state === "running";
  const allPagesDone = item.total_pages > 0 && item.processed_pages >= item.total_pages;
  const percent = item.total_pages > 0 ? Math.round((100 * item.processed_pages) / item.total_pages) : item.progress_percent;
  const canOpen = item.state === "completed" || item.state === "duplicate" || item.state === "running";

  return (
    <li className="px-4 py-3" data-testid={`batch-row-${item.job_id}`}>
      <div className="flex items-start gap-3">
        <span className="text-slate-400 w-6 text-right pt-0.5">{index}.</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-slate-900 truncate">{item.filename}</span>
            <span className={clsx("text-xs px-2 py-0.5 rounded-full", pill.className)}>{pill.label}</span>
          </div>
          <div className="text-xs text-slate-500 mt-0.5">
            {pageWord(item.total_pages)} · {formatBytes(item.file_size_bytes)}
            {item.duration_seconds !== null && item.state === "completed" && ` · ${formatElapsed(item.duration_seconds)}`}
            {item.attempts > 1 && ` · attempt ${item.attempts}`}
          </div>

          {showProgress && (
            <div className="mt-2 max-w-md">
              <ProgressBar
                percent={percent}
                label={allPagesDone ? "Finalizing exports…" : `${item.processed_pages} / ${item.total_pages} pages`}
              />
            </div>
          )}

          {item.state === "failed" && item.error_message && (
            <div className="mt-2 text-xs text-red-700 bg-red-50 rounded px-2 py-1">{item.error_message}</div>
          )}

          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
            {canOpen && (
              <Link className="text-brand-600 hover:underline" to={`/documents/${item.document_id}`}>
                {item.state === "running" ? "Open" : "View / edit"}
              </Link>
            )}
            {(item.state === "completed" || (item.state === "duplicate" && item.status === "COMPLETED")) && (
              <>
                <a className="text-brand-600 hover:underline" href={documentsApi.downloadPdfUrl(item.document_id)}>
                  Searchable PDF
                </a>
                <a className="text-brand-600 hover:underline" href={documentsApi.downloadDocxUrl(item.document_id)}>
                  DOCX
                </a>
                <a className="text-brand-600 hover:underline" href={documentsApi.downloadExcelUrl(item.document_id)}>
                  Excel
                </a>
              </>
            )}
            {(item.state === "failed" || item.state === "cancelled") && (
              <button
                className="text-brand-600 hover:underline disabled:opacity-40"
                disabled={retrying}
                onClick={() => onRetry(item.job_id)}
              >
                Retry
              </button>
            )}
          </div>
        </div>
      </div>
    </li>
  );
}

function Summary({ batch }: { batch: Batch }) {
  const done = batch.status !== "processing" && batch.status !== "uploading";
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-3">
      <div className="flex flex-wrap justify-between gap-2 items-baseline">
        <div className="text-lg font-semibold text-slate-900">
          {batch.completed_files} / {batch.total_files} completed
          {batch.failed_files > 0 && <span className="text-red-700 font-normal text-sm"> · {batch.failed_files} failed</span>}
          {batch.cancelled_files > 0 && (
            <span className="text-slate-500 font-normal text-sm"> · {batch.cancelled_files} cancelled</span>
          )}
        </div>
        {batch.elapsed_seconds !== null && (
          <div className="text-sm text-slate-500">
            {done ? "Total time" : "Elapsed"} {formatElapsed(batch.elapsed_seconds)}
          </div>
        )}
      </div>
      <ProgressBar percent={batch.overall_percent} label={`${batch.processed_pages} / ${batch.total_pages} pages`} />
      <div className="text-xs text-slate-500">
        {batch.ocr_provider} · {batch.dpi} DPI · {batch.preprocess_profile.toLowerCase().replace("_", " ")}
        {batch.concurrency && !done && (
          <>
            {" "}
            · processing up to {batch.concurrency.limit} file{batch.concurrency.limit === 1 ? "" : "s"} at a time on this server (
            {batch.concurrency.limiting_factor}-limited); smaller files go first
          </>
        )}
      </div>
    </div>
  );
}

export default function BatchPage() {
  const { batchId = "" } = useParams<{ batchId: string }>();
  const location = useLocation();
  const skipped = ((location.state as { skipped?: SkippedFile[] } | null)?.skipped ?? []) as SkippedFile[];

  const { data: batch, refetch, error } = useQuery({
    queryKey: ["batch", batchId],
    queryFn: () => batchApi.status(batchId),
    enabled: !!batchId,
    // Light polling: 3s while a file is actually being processed, 5s while
    // only waiting, none once everything is finished. (react-query also
    // pauses it while the tab is in the background.)
    refetchInterval: (q) => {
      const d = q.state.data;
      if (!d) return 3000;
      if (d.status !== "processing" && d.status !== "uploading") return false;
      return d.processing_files > 0 ? 3000 : 5000;
    },
  });

  const retry = useMutation({
    mutationFn: (jobId: string) => batchApi.retry(batchId, jobId),
    onSuccess: () => refetch(),
  });
  const cancel = useMutation({ mutationFn: () => batchApi.cancel(batchId), onSuccess: () => refetch() });

  if (error && !batch) {
    return (
      <div className="max-w-3xl mx-auto space-y-4">
        <div className="rounded-lg bg-red-50 text-red-700 text-sm px-4 py-3">Could not load this batch.</div>
        <Link to="/upload" className="text-brand-600 hover:underline text-sm">
          ← Back to upload
        </Link>
      </div>
    );
  }
  if (!batch) return <div className="text-slate-500">Loading…</div>;

  const active = batch.status === "processing";

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Multiple OCR Queue</h1>
          <p className="text-slate-500 mt-1">Each file is a normal document — open it any time, even while others are still running.</p>
        </div>
        <div className="flex gap-2">
          {active && (
            <button
              onClick={() => cancel.mutate()}
              disabled={cancel.isPending}
              className="px-3 py-2 rounded-lg border border-slate-300 text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-40"
            >
              Cancel remaining
            </button>
          )}
          <Link to="/upload" className="px-3 py-2 rounded-lg bg-brand-600 text-white text-sm hover:bg-brand-700">
            New upload
          </Link>
        </div>
      </div>

      {skipped.length > 0 && (
        <div className="rounded-lg bg-amber-50 text-amber-900 text-sm px-4 py-3">
          <div className="font-medium">
            {skipped.length} file{skipped.length === 1 ? " was" : "s were"} not added:
          </div>
          <ul className="list-disc ml-5 mt-1">
            {skipped.map((s) => (
              <li key={s.name}>
                {s.name} — {s.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      <Summary batch={batch} />

      <ul className="bg-white rounded-xl border border-slate-200 divide-y divide-slate-100 text-sm">
        {batch.items.map((item, i) => (
          <FileRow key={item.job_id} index={i + 1} item={item} onRetry={(id) => retry.mutate(id)} retrying={retry.isPending} />
        ))}
      </ul>
    </div>
  );
}
