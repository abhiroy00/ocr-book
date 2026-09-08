import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { documentsApi, fileUrl } from "@/services/api";
import PageOverlay from "@/components/PageOverlay";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import type { LayoutBlockRecord, TableRecord } from "@/types/document";

/**
 * Document editor (spec section 22/23): click a text block to edit it,
 * click a table to edit cells (add/delete row/column, merge). Edits never
 * silently overwrite OCR source text elsewhere — only the block/cell the
 * user explicitly changes is updated.
 */
export default function DocumentEditPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const pageNumber = Number(searchParams.get("page") || 1);
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);
  const [selectedTableId, setSelectedTableId] = useState<string | null>(null);

  const { data: pages } = useQuery({ queryKey: ["pages", id], queryFn: () => documentsApi.pages(id!), enabled: !!id });
  const { data: layoutBlocks } = useQuery({
    queryKey: ["layout", id, pageNumber],
    queryFn: () => documentsApi.layoutBlocks(id!, pageNumber),
    enabled: !!id,
  });
  const { data: tables } = useQuery({
    queryKey: ["tables", id, pageNumber],
    queryFn: () => documentsApi.tables(id!, pageNumber),
    enabled: !!id,
  });
  const { data: ocrBlocks } = useQuery({
    queryKey: ["ocr", id, pageNumber],
    queryFn: () => documentsApi.ocrBlocks(id!, pageNumber),
    enabled: !!id,
  });

  const page = pages?.find((p) => p.page_number === pageNumber);
  const totalPages = pages?.length ?? 0;
  const goToPage = (n: number) => {
    setSelectedBlockId(null);
    setSelectedTableId(null);
    setSearchParams({ page: String(Math.min(Math.max(1, n), totalPages || 1)) });
  };

  const selectedBlock = layoutBlocks?.find((b) => b.id === selectedBlockId) ?? null;
  const selectedTable = tables?.find((t) => t.id === selectedTableId) ?? null;
  const lowConfidenceOcr = (ocrBlocks ?? []).filter((b) => b.confidence < 0.8);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Link to={`/documents/${id}`} className="text-sm text-slate-500 hover:text-slate-700">
            ← Back
          </Link>
          <h1 className="text-xl font-semibold text-slate-900 ml-2">Editor</h1>
        </div>
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
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 bg-white rounded-xl border border-slate-200 overflow-auto max-h-[75vh] p-4">
          {page && layoutBlocks && (
            <div className="relative inline-block" style={{ width: page.width }}>
              <img src={fileUrl(page.processed_image_url || page.original_image_url)} alt="" className="w-full h-auto block" draggable={false} />
              <ClickableOverlay
                pageWidth={page.width}
                pageHeight={page.height}
                blocks={layoutBlocks}
                tables={tables ?? []}
                selectedId={selectedBlockId || selectedTableId}
                onSelectBlock={(b) => {
                  setSelectedTableId(null);
                  setSelectedBlockId(b.id);
                }}
                onSelectTable={(t) => {
                  setSelectedBlockId(null);
                  setSelectedTableId(t.id);
                }}
              />
            </div>
          )}
        </div>

        <div className="space-y-4">
          {selectedBlock && <BlockEditor documentId={id!} block={selectedBlock} onClose={() => setSelectedBlockId(null)} />}
          {selectedTable && <TableEditor documentId={id!} table={selectedTable} onClose={() => setSelectedTableId(null)} />}
          {!selectedBlock && !selectedTable && (
            <div className="bg-white rounded-xl border border-slate-200 p-5 text-sm text-slate-500">
              Click a text block or table on the page to edit it.
            </div>
          )}

          {lowConfidenceOcr.length > 0 && (
            <div className="bg-white rounded-xl border border-amber-200 p-5">
              <h3 className="font-medium text-slate-900 mb-2">Review OCR ({lowConfidenceOcr.length})</h3>
              <p className="text-xs text-slate-500 mb-3">
                Low-confidence OCR is never auto-corrected — review and edit the containing text block above.
              </p>
              <div className="space-y-2 max-h-64 overflow-auto">
                {lowConfidenceOcr.map((b) => (
                  <div key={b.id} className="flex items-center justify-between text-xs border border-slate-100 rounded-md px-2 py-1.5">
                    <span className="truncate mr-2">{b.text || <em className="text-slate-400">(empty)</em>}</span>
                    <ConfidenceBadge value={b.confidence} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ClickableOverlay({
  pageWidth,
  pageHeight,
  blocks,
  tables,
  selectedId,
  onSelectBlock,
  onSelectTable,
}: {
  pageWidth: number;
  pageHeight: number;
  blocks: LayoutBlockRecord[];
  tables: TableRecord[];
  selectedId: string | null;
  onSelectBlock: (b: LayoutBlockRecord) => void;
  onSelectTable: (t: TableRecord) => void;
}) {
  const boxes = [
    ...blocks
      .filter((b) => b.block_type !== "table")
      .map((b) => ({ id: b.id, bbox: b.bbox, kind: "layout" as const, onClick: () => onSelectBlock(b) })),
    ...tables.map((t) => ({ id: t.id, bbox: t.bbox, kind: "table" as const, onClick: () => onSelectTable(t) })),
  ];
  return (
    <>
      <PageOverlay pageWidth={pageWidth} pageHeight={pageHeight} boxes={boxes} />
      {selectedId && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${pageWidth} ${pageHeight}`} preserveAspectRatio="none">
          {[...blocks, ...tables]
            .filter((b) => b.id === selectedId)
            .map((b) => (
              <rect
                key={b.id}
                x={b.bbox.x1}
                y={b.bbox.y1}
                width={b.bbox.x2 - b.bbox.x1}
                height={b.bbox.y2 - b.bbox.y1}
                fill="rgba(49,130,246,0.15)"
                stroke="#3182f6"
                strokeWidth={pageWidth / 250}
              />
            ))}
        </svg>
      )}
    </>
  );
}

function BlockEditor({ documentId, block, onClose }: { documentId: string; block: LayoutBlockRecord; onClose: () => void }) {
  const queryClient = useQueryClient();
  const [text, setText] = useState(block.content?.text ?? "");

  const mutation = useMutation({
    mutationFn: (newText: string) => documentsApi.updateBlock(documentId, block.id, { text: newText }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["layout", documentId] }),
  });

  return (
    <div className="bg-white rounded-xl border border-brand-300 p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-slate-900 capitalize">{block.block_type.replaceAll("_", " ")}</h3>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-sm">
          ✕
        </button>
      </div>
      <div className="text-xs text-slate-500 mb-2">
        Confidence: <ConfidenceBadge value={block.confidence} />
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm font-mono"
      />
      <button
        onClick={() => mutation.mutate(text)}
        disabled={mutation.isPending}
        className="mt-3 w-full py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700 disabled:opacity-50"
      >
        {mutation.isPending ? "Saving…" : "Save"}
      </button>
    </div>
  );
}

function TableEditor({ documentId, table, onClose }: { documentId: string; table: TableRecord; onClose: () => void }) {
  const queryClient = useQueryClient();
  const [selectedCells, setSelectedCells] = useState<string[]>([]);

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["tables", documentId] });

  const cellMutation = useMutation({
    mutationFn: ({ cellId, text }: { cellId: string; text: string }) => documentsApi.updateCell(documentId, cellId, { text }),
    onSuccess: invalidate,
  });

  const structureMutation = useMutation({
    mutationFn: (body: { op: string; row?: number; column?: number; cell_ids?: string[] }) =>
      documentsApi.updateTableStructure(documentId, table.id, body),
    onSuccess: () => {
      invalidate();
      setSelectedCells([]);
    },
  });

  // The backend only ever stores one row/column *origin* per merged cell
  // (see document_service._merge_cells) — so grouping by row and rendering
  // each cell with its rowSpan/colSpan is enough; the browser handles the
  // spanning for cells covered by an earlier row/column automatically. A
  // naive full (row, col) grid walk would re-render the same merged cell
  // at every position its span covers, producing duplicate <td>s.
  const rows: (typeof table.cells)[] = Array.from({ length: table.n_rows }, () => []);
  for (const cell of table.cells) {
    if (rows[cell.row]) rows[cell.row].push(cell);
  }
  rows.forEach((row) => row.sort((a, b) => a.column - b.column));

  const toggleCellSelection = (cellId: string) =>
    setSelectedCells((prev) => (prev.includes(cellId) ? prev.filter((id) => id !== cellId) : [...prev, cellId]));

  return (
    <div className="bg-white rounded-xl border border-emerald-300 p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-slate-900">
          Table ({table.n_rows}×{table.n_cols})
        </h3>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-sm">
          ✕
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mb-3 text-xs">
        <button onClick={() => structureMutation.mutate({ op: "add_row" })} className="px-2 py-1 rounded border border-slate-200 hover:bg-slate-50">
          + Row
        </button>
        <button onClick={() => structureMutation.mutate({ op: "add_column" })} className="px-2 py-1 rounded border border-slate-200 hover:bg-slate-50">
          + Column
        </button>
        <button
          disabled={selectedCells.length < 2}
          onClick={() => structureMutation.mutate({ op: "merge_cells", cell_ids: selectedCells })}
          className="px-2 py-1 rounded border border-slate-200 hover:bg-slate-50 disabled:opacity-40"
        >
          Merge selected
        </button>
      </div>

      <div className="overflow-auto max-h-96 border border-slate-200 rounded-lg">
        <table className="text-xs border-collapse w-full">
          <tbody>
            {rows.map((row, rIdx) => (
              <tr key={rIdx}>
                {row.map((cell) => (
                  <TableCellInput
                    key={cell.id}
                    cell={cell}
                    selected={selectedCells.includes(cell.id)}
                    onToggleSelect={() => toggleCellSelection(cell.id)}
                    onSave={(text) => cellMutation.mutate({ cellId: cell.id, text })}
                  />
                ))}
                <td className="border-none pl-1">
                  <button
                    title="Delete row"
                    onClick={() => structureMutation.mutate({ op: "delete_row", row: rIdx })}
                    className="text-slate-300 hover:text-red-500"
                  >
                    ✕
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TableCellInput({
  cell,
  selected,
  onToggleSelect,
  onSave,
}: {
  cell: { id: string; text: string; is_header: boolean; align_h: string; rowspan: number; colspan: number };
  selected: boolean;
  onToggleSelect: () => void;
  onSave: (text: string) => void;
}) {
  const [value, setValue] = useState(cell.text);
  return (
    <td
      rowSpan={cell.rowspan}
      colSpan={cell.colspan}
      className={`border border-slate-200 p-0 ${selected ? "ring-2 ring-brand-500" : ""} ${cell.is_header ? "bg-slate-50 font-medium" : ""}`}
    >
      <div className="flex items-stretch">
        <input type="checkbox" checked={selected} onChange={onToggleSelect} className="ml-1 self-center" />
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onBlur={() => value !== cell.text && onSave(value)}
          style={{ textAlign: cell.align_h.toLowerCase() as any }}
          className="w-full px-2 py-1.5 outline-none focus:bg-brand-50"
        />
      </div>
    </td>
  );
}
