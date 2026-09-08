import clsx from "clsx";

/**
 * Confidence tiering per spec section 23:
 *   >95%   normal
 *   80-95% warning
 *   <80%   needs review
 */
export default function ConfidenceBadge({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) {
    return <span className="text-xs text-slate-400">—</span>;
  }
  const pct = Math.round(value * 100);
  const tier = value > 0.95 ? "high" : value >= 0.8 ? "warning" : "low";
  return (
    <span
      className={clsx("inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium", {
        "bg-emerald-100 text-emerald-800": tier === "high",
        "bg-amber-100 text-amber-800": tier === "warning",
        "bg-red-100 text-red-800": tier === "low",
      })}
      title={tier === "low" ? "Needs review" : undefined}
    >
      {pct}%{tier === "low" && " ⚠"}
    </span>
  );
}
