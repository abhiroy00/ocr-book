import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { accessionApi, documentsApi } from "@/services/api";
import StatusBadge from "@/components/StatusBadge";

export default function DashboardPage() {
  const { data: stats } = useQuery({ queryKey: ["stats"], queryFn: documentsApi.stats, refetchInterval: 5000 });
  const { data: recent } = useQuery({
    queryKey: ["documents", { page: 1, page_size: 5 }],
    queryFn: () => documentsApi.list({ page: 1, page_size: 5 }),
    refetchInterval: 5000,
  });
  const { data: accessionSummary } = useQuery({
    queryKey: ["accessionSummary"],
    queryFn: accessionApi.summary,
    refetchInterval: 5000,
  });

  const cards = [
    { label: "Total documents", value: stats?.total ?? "—", accent: "text-slate-900" },
    { label: "Processing", value: stats?.processing ?? "—", accent: "text-amber-600" },
    { label: "Completed", value: stats?.completed ?? "—", accent: "text-emerald-600" },
    { label: "Failed", value: stats?.failed ?? "—", accent: "text-red-600" },
  ];

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Dashboard</h1>
          <p className="text-slate-500 mt-1">Overview of your scanned document processing pipeline.</p>
        </div>
        <Link to="/upload" className="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700">
          Upload document
        </Link>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {cards.map((c) => (
          <div key={c.label} className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="text-sm text-slate-500">{c.label}</div>
            <div className={`text-3xl font-semibold mt-2 ${c.accent}`}>{c.value}</div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-xl border border-slate-200">
        <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
          <h2 className="font-semibold text-slate-900">Recent documents</h2>
          <Link to="/documents" className="text-sm text-brand-600 hover:underline">
            View all
          </Link>
        </div>
        <div className="divide-y divide-slate-100">
          {recent?.items.length ? (
            recent.items.map((doc) => (
              <Link
                key={doc.id}
                to={`/documents/${doc.id}`}
                className="flex items-center justify-between px-5 py-3 hover:bg-slate-50 transition-colors"
              >
                <div>
                  <div className="font-medium text-slate-900">{doc.original_filename}</div>
                  <div className="text-xs text-slate-500">
                    {doc.page_count} page{doc.page_count === 1 ? "" : "s"} · {new Date(doc.created_at).toLocaleString()}
                  </div>
                </div>
                <StatusBadge status={doc.status} />
              </Link>
            ))
          ) : (
            <div className="px-5 py-8 text-center text-slate-400 text-sm">No documents yet — upload your first scan.</div>
          )}
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-center justify-between flex-wrap gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Library accession register</h2>
          <p className="text-sm text-slate-500 mt-1">
            {accessionSummary
              ? `${accessionSummary.total_records} record${accessionSummary.total_records === 1 ? "" : "s"} from ${accessionSummary.total_documents_processed} document${accessionSummary.total_documents_processed === 1 ? "" : "s"}`
              : "Cumulative bibliographic data extracted from every processed document."}
            {!!accessionSummary?.needs_review_count && (
              <span className="text-amber-600"> · {accessionSummary.needs_review_count} need review</span>
            )}
          </p>
        </div>
        <a
          href={accessionApi.masterExcelUrl()}
          className="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700"
        >
          Download Master Excel
        </a>
      </div>
    </div>
  );
}
