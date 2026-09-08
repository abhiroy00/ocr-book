import type { BBox, LayoutBlockType } from "@/types/document";

interface OverlayBox {
  id: string;
  bbox: BBox;
  kind: "ocr" | "layout" | "table";
  label?: string;
  blockType?: LayoutBlockType;
  onClick?: () => void;
}

const KIND_COLOR: Record<OverlayBox["kind"], string> = {
  ocr: "#f59e0b",
  layout: "#3182f6",
  table: "#16a34a",
};

/**
 * Renders bounding-box overlays over a page image using an SVG whose
 * viewBox matches the page's pixel coordinate system — so boxes always
 * align with the image regardless of the image's displayed (zoomed) size.
 */
export default function PageOverlay({
  pageWidth,
  pageHeight,
  boxes,
}: {
  pageWidth: number;
  pageHeight: number;
  boxes: OverlayBox[];
}) {
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox={`0 0 ${pageWidth} ${pageHeight}`}
      preserveAspectRatio="none"
    >
      {boxes.map((box) => (
        <g key={box.id}>
          <rect
            x={box.bbox.x1}
            y={box.bbox.y1}
            width={Math.max(0, box.bbox.x2 - box.bbox.x1)}
            height={Math.max(0, box.bbox.y2 - box.bbox.y1)}
            fill="none"
            stroke={KIND_COLOR[box.kind]}
            strokeWidth={Math.max(1, pageWidth / 500)}
            className={box.onClick ? "pointer-events-auto cursor-pointer hover:fill-brand-500/10" : undefined}
            onClick={box.onClick}
          />
        </g>
      ))}
    </svg>
  );
}
