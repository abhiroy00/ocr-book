import clsx from "clsx";
import type { DocumentStatus } from "@/types/document";

const STYLES: Record<DocumentStatus, string> = {
  UPLOADED: "bg-slate-100 text-slate-700",
  QUEUED: "bg-slate-100 text-slate-700",
  PROCESSING: "bg-amber-100 text-amber-800",
  OCR_PROCESSING: "bg-amber-100 text-amber-800",
  LAYOUT_PROCESSING: "bg-amber-100 text-amber-800",
  TABLE_PROCESSING: "bg-amber-100 text-amber-800",
  RECONSTRUCTING: "bg-amber-100 text-amber-800",
  EXPORTING: "bg-amber-100 text-amber-800",
  COMPLETED: "bg-emerald-100 text-emerald-800",
  FAILED: "bg-red-100 text-red-800",
  CANCELLED: "bg-slate-200 text-slate-600",
};

export default function StatusBadge({ status }: { status: DocumentStatus }) {
  return (
    <span className={clsx("inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium", STYLES[status])}>
      {status.replaceAll("_", " ")}
    </span>
  );
}
