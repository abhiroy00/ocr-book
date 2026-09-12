import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { documentsApi } from "@/services/api";
import ConfidenceBadge from "@/components/ConfidenceBadge";

export default function DocumentExportPage() {
  const { id } = useParams<{ id: string }>();
  const { data: document } = useQuery({ queryKey: ["document", id], queryFn: () => documentsApi.get(id!), enabled: !!id });
  const { data: quality } = useQuery({ queryKey: ["quality", id], queryFn: () => documentsApi.quality(id!), enabled: !!id });

  const searchablePdfMutation = useMutation({ mutationFn: () => documentsApi.exportSearchablePdf(id!) });
  const excelMutation = useMutation({ mutationFn: () => documentsApi.exportExcel(id!) });

  if (!document) return <div className="text-slate-400 text-sm">Loading…</div>;

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex items-center gap-2">
        <Link to={`/documents/${id}`} className="text-sm text-slate-500 hover:text-slate-700">
          ← Back
        </Link>
        <h1 className="text-xl font-semibold text-slate-900 ml-2">Export — {document.original_filename}</h1>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
        <h2 className="font-semibold text-slate-900">Cleaned Searchable PDF</h2>
        <p className="text-sm text-slate-500">
          The original book's page images, cleaned, with an invisible OCR text layer — looks exactly like the source book (same layout,
          tables, images) but is now searchable and copy-pasteable (Ctrl+F works, including Hindi). This is the visual-fidelity output;
          it is not rebuilt from fonts/text boxes.
        </p>
        <div className="flex gap-3">
          <button
            onClick={() => searchablePdfMutation.mutate()}
            disabled={searchablePdfMutation.isPending}
            className="px-4 py-2 rounded-lg border border-slate-200 text-sm font-medium hover:bg-slate-50 disabled:opacity-50"
          >
            {searchablePdfMutation.isPending ? "Regenerating…" : "Regenerate from current pages"}
          </button>
          <a
            href={documentsApi.downloadPdfUrl(id!)}
            className="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700"
          >
            Download PDF
          </a>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
        <h2 className="font-semibold text-slate-900">Structured Excel</h2>
        <p className="text-sm text-slate-500">
          Every detected table as its own sheet with real rows/columns (not one giant OCR-text cell), plus a Table_Index sheet linking
          each table back to its source page and detection confidence. Numbers are kept exactly as OCR'd — never reformatted or
          "corrected".
        </p>
        <div className="flex gap-3">
          <button
            onClick={() => excelMutation.mutate()}
            disabled={excelMutation.isPending}
            className="px-4 py-2 rounded-lg border border-slate-200 text-sm font-medium hover:bg-slate-50 disabled:opacity-50"
          >
            {excelMutation.isPending ? "Generating…" : "Regenerate Excel"}
          </button>
          <a
            href={documentsApi.downloadExcelUrl(id!)}
            className="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700"
          >
            Download Excel
          </a>
        </div>
      </div>

      {quality && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-slate-900">Quality report</h2>
            <span className="text-sm text-slate-500">
              Overall visual similarity: <ConfidenceBadge value={quality.overall_visual_similarity} />
            </span>
          </div>
          <div className="overflow-auto max-h-72">
            <table className="w-full text-xs">
              <thead className="text-slate-500 text-left">
                <tr>
                  <th className="py-1.5 pr-3">Page</th>
                  <th className="py-1.5 pr-3">OCR</th>
                  <th className="py-1.5 pr-3">Layout</th>
                  <th className="py-1.5 pr-3">Table</th>
                  <th className="py-1.5 pr-3">Similarity</th>
                  <th className="py-1.5">Review</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {quality.pages.map((p) => (
                  <tr key={p.page_number}>
                    <td className="py-1.5 pr-3 font-medium">{p.page_number}</td>
                    <td className="py-1.5 pr-3">
                      <ConfidenceBadge value={p.ocr_confidence_avg} />
                    </td>
                    <td className="py-1.5 pr-3">
                      <ConfidenceBadge value={p.layout_confidence_avg} />
                    </td>
                    <td className="py-1.5 pr-3">
                      <ConfidenceBadge value={p.table_confidence_avg} />
                    </td>
                    <td className="py-1.5 pr-3">
                      <ConfidenceBadge value={p.visual_similarity_score} />
                    </td>
                    <td className="py-1.5">{p.needs_review && <span className="text-amber-600">⚠ Needs review</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
