import { useEffect, useState } from "react";

/** `123` -> `"02:03"`, `4000` -> `"01:06:40"`. Always HH:MM:SS once an
 * hour is crossed, MM:SS below that (spec: MM:SS acceptable under an
 * hour, HH:MM:SS preferred once it matters) -- padded to two digits
 * throughout so the display never jumps in width as digits roll over. */
function formatDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return hh > 0 ? `${pad(hh)}:${pad(mm)}:${pad(ss)}` : `${pad(mm)}:${pad(ss)}`;
}

function formatWords(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  if (mins === 0) return `${secs} second${secs === 1 ? "" : "s"}`;
  return `${mins} minute${mins === 1 ? "" : "s"} ${secs} second${secs === 1 ? "" : "s"}`;
}

/**
 * Live processing-time display for the Processing panel.
 *
 * Source of truth is always the backend: while a job is running, the
 * ticking number is `now - startedAt` (a purely visual approximation,
 * recomputed every second, that also transparently "resumes" correctly
 * across a page refresh since `startedAt` is re-fetched from the server
 * rather than kept in any local/frontend-only state). Once the job
 * reaches a terminal state, the timer freezes on `durationSeconds` -- the
 * backend-computed, authoritative final value -- never a number derived
 * from frontend timestamps.
 *
 * Renders nothing until `startedAt` is set: the timer must start when the
 * backend confirms processing actually started (the first PROCESSING
 * transition -- see `document_service.update_job_progress`), not merely
 * when this component mounts/the page loads.
 */
export default function ProcessTimer({
  startedAt,
  finishedAt,
  durationSeconds,
}: {
  startedAt: string | null;
  finishedAt: string | null;
  durationSeconds: number | null;
}) {
  const isRunning = !!startedAt && !finishedAt;
  const [liveElapsedSec, setLiveElapsedSec] = useState<number>(() =>
    startedAt ? (Date.now() - new Date(startedAt).getTime()) / 1000 : 0,
  );

  useEffect(() => {
    if (!isRunning || !startedAt) return;
    const startMs = new Date(startedAt).getTime();
    // Set immediately (don't wait a full second for the first tick --
    // matters most right after a page refresh) and then once per second.
    const tick = () => setLiveElapsedSec((Date.now() - startMs) / 1000);
    tick();
    const intervalId = window.setInterval(tick, 1000);
    return () => window.clearInterval(intervalId);
    // Only re-arm the interval if the job identity actually changes
    // (a different start time) or it stops running -- never on every
    // parent re-render, which would otherwise stack up extra intervals.
  }, [isRunning, startedAt]);

  if (!startedAt) return null;

  const finalSeconds = durationSeconds ?? (finishedAt ? (new Date(finishedAt).getTime() - new Date(startedAt).getTime()) / 1000 : null);
  const displaySeconds = finalSeconds ?? liveElapsedSec;

  return (
    <div className="flex items-center gap-2">
      {finalSeconds !== null ? (
        <span className="text-emerald-600" aria-hidden="true">
          ✓
        </span>
      ) : (
        <span className="relative flex h-2 w-2" aria-hidden="true">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-500" />
        </span>
      )}
      <div>
        <div className="text-xs text-slate-500">{finalSeconds !== null ? "Total Processing Time" : "Processing Time"}</div>
        <div className="text-sm font-mono font-semibold text-slate-900 tabular-nums">
          {formatDuration(displaySeconds)}
          <span className="ml-2 font-sans font-normal text-xs text-slate-400">({formatWords(displaySeconds)})</span>
        </div>
      </div>
    </div>
  );
}
