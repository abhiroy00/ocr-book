export default function ProgressBar({ percent, label }: { percent: number; label?: string }) {
  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>{label}</span>
          <span>{percent}%</span>
        </div>
      )}
      <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
        <div
          className="h-full bg-brand-600 transition-all duration-300 ease-out"
          style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
        />
      </div>
    </div>
  );
}
