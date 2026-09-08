import { useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, useSearchParams, Link } from "react-router-dom";
import { documentsApi, fileUrl } from "@/services/api";
import { useViewerStore } from "@/store/viewerStore";
import PageOverlay from "@/components/PageOverlay";
import ConfidenceBadge from "@/components/ConfidenceBadge";

/**
 * Before/After preview (spec section 21 — mandatory): original scan on the
 * left, cleaned/reconstructed page on the right, with zoom, page nav,
 * synchronized scrolling, and toggleable OCR/layout/table overlays.
 */
export default function DocumentPreviewPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const pageNumber = Number(searchParams.get("page") || 1);

  const { zoom, setZoom, syncScroll, toggleSyncScroll, showOcrBoxes, showLayoutBoxes, showTableBoxes, toggleOcrBoxes, toggleLayoutBoxes, toggleTableBoxes } =
    useViewerStore();

  const { data: pages } = useQuery({ queryKey: ["pages", id], queryFn: () => documentsApi.pages(id!), enabled: !!id });
  const { data: ocrBlocks } = useQuery({
    queryKey: ["ocr", id, pageNumber],
    queryFn: () => documentsApi.ocrBlocks(id!, pageNumber),
    enabled: !!id && showOcrBoxes,
  });
  const { data: layoutBlocks } = useQuery({
    queryKey: ["layout", id, pageNumber],
    queryFn: () => documentsApi.layoutBlocks(id!, pageNumber),
    enabled: !!id && showLayoutBoxes,
  });
  const { data: tables } = useQuery({
    queryKey: ["tables", id, pageNumber],
    queryFn: () => documentsApi.tables(id!, pageNumber),
    enabled: !!id && showTableBoxes,
  });

  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const syncing = useRef(false);

  const onScroll = (source: "left" | "right") => (e: React.UIEvent<HTMLDivElement>) => {
    if (!syncScroll || syncing.current) return;
    const other = source === "left" ? rightRef.current : leftRef.current;
    if (!other) return;
    syncing.current = true;
    other.scrollTop = e.currentTarget.scrollTop;
    other.scrollLeft = e.currentTarget.scrollLeft;
    syncing.current = false;
  };

  const page = pages?.find((p) => p.page_number === pageNumber);
  const totalPages = pages?.length ?? 0;

  const goToPage = (n: number) => setSearchParams({ page: String(Math.min(Math.max(1, n), totalPages || 1)) });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <Link to={`/documents/${id}`} className="text-sm text-slate-500 hover:text-slate-700">
            ← Back
          </Link>
          <h1 className="text-xl font-semibold text-slate-900 ml-2">Before / After</h1>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 bg-white border border-slate-200 rounded-lg px-2 py-1">
            <button onClick={() => goToPage(pageNumber - 1)} disabled={pageNumber <= 1} className="px-2 disabled:opacity-30">
              ‹
            </button>
            <span className="text-sm px-2">
              Page {pageNumber} / {totalPages || "…"}
            </span>
            <button onClick={() => goToPage(pageNumber + 1)} disabled={pageNumber >= totalPages} className="px-2 disabled:opacity-30">
              ›
            </button>
          </div>

          <div className="flex items-center gap-1 bg-white border border-slate-200 rounded-lg px-2 py-1">
            <button onClick={() => setZoom(zoom - 0.25)} className="px-2">
              −
            </button>
            <span className="text-sm px-2 w-14 text-center">{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom(zoom + 0.25)} className="px-2">
              +
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-sm">
        <ToggleChip active={showOcrBoxes} onClick={toggleOcrBoxes} color="#f59e0b" label="OCR boxes" />
        <ToggleChip active={showLayoutBoxes} onClick={toggleLayoutBoxes} color="#3182f6" label="Layout blocks" />
        <ToggleChip active={showTableBoxes} onClick={toggleTableBoxes} color="#16a34a" label="Table detection" />
        <ToggleChip active={syncScroll} onClick={toggleSyncScroll} color="#64748b" label="Sync scroll" />
      </div>

      {page && (
        <div className="flex items-center gap-4 text-xs text-slate-500">
          <span>
            OCR confidence: <ConfidenceBadge value={page.ocr_confidence_avg} />
          </span>
          <span>
            Layout confidence: <ConfidenceBadge value={page.layout_confidence_avg} />
          </span>
          <span>
            Table confidence: <ConfidenceBadge value={page.table_confidence_avg} />
          </span>
          <span>
            Visual similarity: <ConfidenceBadge value={page.visual_similarity_score} />
          </span>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 h-[75vh]">
        <PanelViewer
          title="Original scan"
          imageUrl={page ? fileUrl(page.original_image_url) : undefined}
          pageWidth={page?.width ?? 0}
          pageHeight={page?.height ?? 0}
          zoom={zoom}
          scrollRef={leftRef}
          onScroll={onScroll("left")}
        />
        <PanelViewer
          title="Cleaned / reconstructed"
          imageUrl={page ? fileUrl(page.processed_image_url || page.original_image_url) : undefined}
          pageWidth={page?.width ?? 0}
          pageHeight={page?.height ?? 0}
          zoom={zoom}
          scrollRef={rightRef}
          onScroll={onScroll("right")}
          overlayBoxes={[
            ...(showOcrBoxes ? (ocrBlocks ?? []).map((b) => ({ id: `ocr-${b.id}`, bbox: b.bbox, kind: "ocr" as const })) : []),
            ...(showLayoutBoxes
              ? (layoutBlocks ?? []).map((b) => ({ id: `layout-${b.id}`, bbox: b.bbox, kind: "layout" as const }))
              : []),
            ...(showTableBoxes ? (tables ?? []).map((t) => ({ id: `table-${t.id}`, bbox: t.bbox, kind: "table" as const })) : []),
          ]}
        />
      </div>
    </div>
  );
}

function ToggleChip({ active, onClick, color, label }: { active: boolean; onClick: () => void; color: string; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-medium ${
        active ? "bg-white border-slate-300 text-slate-900" : "bg-slate-50 border-slate-200 text-slate-400"
      }`}
    >
      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: active ? color : "#cbd5e1" }} />
      {label}
    </button>
  );
}

function PanelViewer({
  title,
  imageUrl,
  pageWidth,
  pageHeight,
  zoom,
  scrollRef,
  onScroll,
  overlayBoxes,
}: {
  title: string;
  imageUrl?: string;
  pageWidth: number;
  pageHeight: number;
  zoom: number;
  scrollRef: React.RefObject<HTMLDivElement>;
  onScroll: (e: React.UIEvent<HTMLDivElement>) => void;
  overlayBoxes?: { id: string; bbox: any; kind: "ocr" | "layout" | "table" }[];
}) {
  return (
    <div className="flex flex-col bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 py-2 border-b border-slate-200 text-sm font-medium text-slate-700">{title}</div>
      <div ref={scrollRef} onScroll={onScroll} className="flex-1 overflow-auto bg-slate-100 p-4">
        {imageUrl ? (
          <div className="relative inline-block" style={{ width: pageWidth * zoom }}>
            <img src={imageUrl} alt={title} className="w-full h-auto block" draggable={false} />
            {overlayBoxes && overlayBoxes.length > 0 && (
              <PageOverlay pageWidth={pageWidth} pageHeight={pageHeight} boxes={overlayBoxes} />
            )}
          </div>
        ) : (
          <div className="text-slate-400 text-sm p-8">Loading…</div>
        )}
      </div>
    </div>
  );
}
