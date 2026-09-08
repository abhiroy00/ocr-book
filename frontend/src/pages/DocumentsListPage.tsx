import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { documentsApi } from "@/services/api";
import StatusBadge from "@/components/StatusBadge";
import type { DocumentStatus } from "@/types/document";

const STATUS_FILTERS: (DocumentStatus | "ALL")[] = ["ALL", "PROCESSING", "COMPLETED", "FAILED"];

export default function DocumentsListPage() {
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState<DocumentStatus | "ALL">("ALL");
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["documents", { page, filter }],
    queryFn: () =>
      documentsApi.list({
        page,
        page_size: 20,
        status: filter === "ALL" ? undefined : filter,
      }),
    refetchInterval: 5000,
  });

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this document? This cannot be undone.")) return;
    await documentsApi.remove(id);
    queryClient.invalidateQueries({ queryKey: ["documents"] });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-slate-900">Documents</h1>
        <Link to="/upload" className="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700">
          Upload document
        </Link>
      </div>

      <div className="flex gap-2">
        {STATUS_FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => {
              setFilter(f);
              setPage(1);
            }}
            className={`px-3 py-1.5 rounded-full text-sm font-medium ${
              filter === f ? "bg-brand-600 text-white" : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
            }`}
          >
            {f === "ALL" ? "All" : f.replaceAll("_", " ")}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-slate-500 text-left">
            <tr>
              <th className="px-5 py-3 font-medium">Filename</th>
              <th className="px-5 py-3 font-medium">Pages</th>
              <th className="px-5 py-3 font-medium">OCR Engine</th>
              <th className="px-5 py-3 font-medium">Status</th>
              <th className="px-5 py-3 font-medium">Uploaded</th>
              <th className="px-5 py-3" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {isLoading && (
              <tr>
                <td colSpan={6} className="px-5 py-8 text-center text-slate-400">
                  Loading…
                </td>
              </tr>
            )}
            {!isLoading && data?.items.length === 0 && (
              <tr>
                <td colSpan={6} className="px-5 py-8 text-center text-slate-400">
                  No documents found.
                </td>
              </tr>
            )}
            {data?.items.map((doc) => (
              <tr key={doc.id} className="hover:bg-slate-50">
                <td className="px-5 py-3">
                  <Link to={`/documents/${doc.id}`} className="font-medium text-slate-900 hover:text-brand-600">
                    {doc.original_filename}
                  </Link>
                </td>
                <td className="px-5 py-3 text-slate-600">{doc.page_count}</td>
                <td className="px-5 py-3 text-slate-600">{doc.ocr_provider}</td>
                <td className="px-5 py-3">
                  <StatusBadge status={doc.status} />
                </td>
                <td className="px-5 py-3 text-slate-500">{new Date(doc.created_at).toLocaleString()}</td>
                <td className="px-5 py-3 text-right">
                  <button onClick={() => handleDelete(doc.id)} className="text-slate-400 hover:text-red-600 text-xs">
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data && data.total > data.page_size && (
        <div className="flex justify-center gap-2">
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
            className="px-3 py-1.5 rounded-lg border border-slate-200 text-sm disabled:opacity-40"
          >
            Previous
          </button>
          <span className="px-3 py-1.5 text-sm text-slate-500">
            Page {page} of {Math.ceil(data.total / data.page_size)}
          </span>
          <button
            disabled={page * data.page_size >= data.total}
            onClick={() => setPage((p) => p + 1)}
            className="px-3 py-1.5 rounded-lg border border-slate-200 text-sm disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
