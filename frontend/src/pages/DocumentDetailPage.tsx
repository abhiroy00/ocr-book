import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { documentsApi, fileUrl } from "@/services/api";
import { useDocumentProgress } from "@/hooks/useDocumentProgress";
import StatusBadge from "@/components/StatusBadge";
import ProgressBar from "@/components/ProgressBar";
import ProcessTimer from "@/components/ProcessTimer";

const STAGE_LABELS: Record<string, { label: string; percent: number }> = {
  upload: { label: "Uploading", percent: 10 },
  render: { label: "Rendering", percent: 20 },
  preprocess: { label: "Cleaning", percent: 30 },
  ocr: { label: "OCR", percent: 50 },
  layout: { label: "Layout Detection", percent: 65 },
  table: { label: "Table Detection", percent: 75 },
  document_json: { label: "Building structure", percent: 80 },
  reconstruct: { label: "Reconstruction", percent: 90 },
  export: { label: "Export", percent: 95 },
  quality: { label: "Quality check", percent: 98 },
  done: { label: "Done", percent: 100 },
};

const ACTIVE_STATUSES = new Set([
  "QUEUED",
  "PROCESSING",
  "OCR_PROCESSING",
  "LAYOUT_PROCESSING",
  "TABLE_PROCESSING",
  "RECONSTRUCTING",
  "EXPORTING",
]);

export default function DocumentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();
  const [cancelRequested, setCancelRequested] = useState(false);
  const { data: document, refetch } = useQuery({
    queryKey: ["document", id],
    queryFn: () => documentsApi.get(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && ACTIVE_STATUSES.has(q.state.data.status) ? 3000 : false),
  });

  const isActive = !!document && ACTIVE_STATUSES.has(document.status);
  const progressEvent = useDocumentProgress(id, isActive);

  // The Process Timer's start/end timestamps live on ProcessingJob, not
  // Document (see backend/app/models/processing_job.py) -- reusing that
  // existing state rather than inventing separate timer bookkeeping.
  // Fetched once for a completed document (so re-opening it still shows
  // its final duration -- test scenario H) and polled at the same 3s
  // cadence as `document`/`pages` while active, which is what actually
  // delivers `started_at` shortly after the backend sets it and
  // `finished_at`/`processing_duration_seconds` the moment the job ends.
  const { data: job, refetch: refetchJob } = useQuery({
    queryKey: ["processingJob", id],
    queryFn: () => documentsApi.progress(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && ACTIVE_STATUSES.has(q.state.data.status) ? 3000 : false),
  });

  // Reset the button's own "Stopping…" flag once the document actually
  // leaves an active status, rather than a fixed timeout -- cancellation
  // is cooperative (the current page finishes first, see
  // `pipeline_tasks._is_cancel_requested`), so how long it takes to land
  // genuinely varies with page size/host load.
  if (cancelRequested && !isActive) {
    setCancelRequested(false);
  }

  const cancelMutation = useMutation({
    mutationFn: () => documentsApi.cancel(id!),
    onSuccess: () => {
      setCancelRequested(true);
      queryClient.invalidateQueries({ queryKey: ["document", id] });
      queryClient.invalidateQueries({ queryKey: ["processingJob", id] });
    },
  });

  if (progressEvent && progressEvent.status !== document?.status) {
    // Refresh the document record once a stage transition lands so the
    // status badge / action buttons stay in sync with the live event.
    refetch();
    // Same idea for the job record -- this is what gets `started_at` in
    // front of the timer promptly (right as PROCESSING begins) and
    // `finished_at`/`processing_duration_seconds` the instant the job
    // ends, rather than waiting for the next 3s poll tick.
    refetchJob();
  }

  if (!document) {
    return <div className="text-slate-400 text-sm">Loading…</div>;
  }

  const currentStage = progressEvent?.stage ?? "upload";
  const currentPercent = progressEvent?.percent ?? (document.status === "COMPLETED" ? 100 : 0);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">{document.original_filename}</h1>
          <div className="flex items-center gap-3 mt-2">
            <StatusBadge status={document.status} />
            <span className="text-sm text-slate-500">
              {document.page_count} pages · {document.dpi} DPI · {document.ocr_provider} · {document.preprocess_profile}
            </span>
            {/* Terminal-state duration (test scenario H: re-opening a
                completed/failed/cancelled document still shows its final
                processing time, sourced from the backend, not recomputed). */}
            {!isActive && job && (
              <ProcessTimer startedAt={job.started_at} finishedAt={job.finished_at} durationSeconds={job.processing_duration_seconds} />
            )}
          </div>
        </div>
        {document.status === "COMPLETED" && (
          <div className="flex gap-2">
            <Link to={`/documents/${id}/preview`} className="px-4 py-2 rounded-lg border border-slate-200 text-sm font-medium hover:bg-slate-50">
              Before / After
            </Link>
            <Link to={`/documents/${id}/edit`} className="px-4 py-2 rounded-lg border border-slate-200 text-sm font-medium hover:bg-slate-50">
              Edit
            </Link>
            <Link to={`/documents/${id}/export`} className="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700">
              Export
            </Link>
          </div>
        )}
      </div>

      {/* Belt-and-suspenders alongside the backend clearing error_message
          on a fresh/successful run: never show a leftover message from an
          earlier failed attempt once the document itself is no longer in
          an error state. */}
      {document.error_message && (document.status === "FAILED" || document.status === "CANCELLED") && (
        <div className="rounded-lg bg-red-50 text-red-700 text-sm px-4 py-3">{document.error_message}</div>
      )}

      {isActive && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
          <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6 space-y-5">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-slate-900">Processing</h2>
              <button
                type="button"
                onClick={() => cancelMutation.mutate()}
                disabled={cancelMutation.isPending || cancelRequested}
                className="px-3 py-1.5 rounded-lg border border-red-200 text-red-700 text-sm font-medium hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {cancelRequested ? "Stopping…" : "Stop"}
              </button>
            </div>
            <ProgressBar percent={currentPercent} label={STAGE_LABELS[currentStage]?.label ?? currentStage} />
            {cancelMutation.isError && (
              <p className="text-xs text-red-600">Could not stop this job. Try again.</p>
            )}
            {job && <ProcessTimer startedAt={job.started_at} finishedAt={job.finished_at} durationSeconds={job.processing_duration_seconds} />}
            {progressEvent?.page && <p className="text-xs text-slate-500">Currently on page {progressEvent.page}</p>}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2">
              {Object.entries(STAGE_LABELS).map(([key, meta]) => {
                const done = currentPercent >= meta.percent;
                const active = key === currentStage;
                return (
                  <div
                    key={key}
                    className={`text-xs px-2 py-1.5 rounded-md border ${
                      active
                        ? "border-brand-500 bg-brand-50 text-brand-700 font-medium"
                        : done
                          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                          : "border-slate-200 text-slate-400"
                    }`}
                  >
                    {meta.label}
                  </div>
                );
              })}
            </div>
          </div>

          <PagesStatusPanel documentId={document.id} pageCount={document.page_count} currentPage={progressEvent?.page ?? null} />
        </div>
      )}

      {document.status === "COMPLETED" && <DocumentPagesGrid documentId={document.id} />}
    </div>
  );
}

/**
 * Live per-page status list shown alongside the overall stage progress bar
 * while a document is processing. A `DocumentPage` row only exists once
 * its render stage has run, so pages beyond that point are shown as
 * "pending" purely from `pageCount` — no page-by-page API call is needed
 * for those.
 */
function PagesStatusPanel({
  documentId,
  pageCount,
  currentPage,
}: {
  documentId: string;
  pageCount: number;
  currentPage: number | null;
}) {
  const { data: pages } = useQuery({
    queryKey: ["pages", documentId],
    queryFn: () => documentsApi.pages(documentId),
    enabled: !!documentId,
    refetchInterval: 3000,
  });

  const pageByNumber = new Map((pages ?? []).map((p) => [p.page_number, p]));

  const statusFor = (n: number): { label: string; className: string } => {
    const page = pageByNumber.get(n);
    if (page && page.ocr_status === "completed" && page.layout_status === "completed" && page.table_status === "completed") {
      return { label: "Done", className: "bg-emerald-50 text-emerald-700 border-emerald-200" };
    }
    if (n === currentPage || page) {
      return { label: "Processing…", className: "bg-brand-50 text-brand-700 border-brand-300" };
    }
    return { label: "Pending", className: "bg-slate-50 text-slate-400 border-slate-200" };
  };

  const doneCount = (pages ?? []).filter(
    (p) => p.ocr_status === "completed" && p.layout_status === "completed" && p.table_status === "completed",
  ).length;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 flex flex-col max-h-[420px]">
      <div className="flex items-center justify-between mb-3 px-1">
        <h3 className="font-medium text-slate-900 text-sm">Pages</h3>
        <span className="text-xs text-slate-500">
          {doneCount} / {pageCount} done
        </span>
      </div>
      <div className="overflow-y-auto space-y-1 pr-1">
        {Array.from({ length: pageCount }, (_, i) => i + 1).map((n) => {
          const s = statusFor(n);
          return (
            <div key={n} className={`flex items-center justify-between text-xs px-2.5 py-1.5 rounded-md border ${s.className}`}>
              <span>Page {n}</span>
              <span className="font-medium">{s.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DocumentPagesGrid({ documentId }: { documentId: string }) {
  const { data: pages } = useQuery({ queryKey: ["pages", documentId], queryFn: () => documentsApi.pages(documentId) });
  if (!pages || pages.length === 0) return null;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6">
      <h2 className="font-semibold text-slate-900 mb-4">Pages ({pages.length})</h2>
      <div className="grid grid-cols-3 sm:grid-cols-6 md:grid-cols-8 gap-3">
        {pages.map((p) => (
          <Link
            key={p.id}
            to={`/documents/${documentId}/preview?page=${p.page_number}`}
            className="border border-slate-200 rounded-lg overflow-hidden hover:border-brand-400 transition-colors"
          >
            <img src={fileUrl(p.processed_image_url || p.original_image_url)} alt={`Page ${p.page_number}`} className="w-full h-28 object-cover bg-slate-50" />
            <div className="text-center text-xs py-1 text-slate-600">Page {p.page_number}</div>
          </Link>
        ))}
      </div>
    </div>
  );
}
