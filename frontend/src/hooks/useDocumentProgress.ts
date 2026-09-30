import { useEffect, useState } from "react";
import axios from "axios";
import { documentsApi, wsUrlForDocument } from "@/services/api";
import type { ProgressEvent } from "@/types/document";

// Missing "CANCELLED" here was a real bug: a stopped job never closed its
// WebSocket / never stopped its polling fallback, both left running for
// the rest of the component's lifetime.
const TERMINAL_STATUSES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);
// The pipeline publishes after every page, so a healthy socket is rarely
// quiet this long outside the slower post-OCR stages.
const SOCKET_SILENCE_MS = 10_000;

/**
 * Live processing progress (spec section 20): prefers the WebSocket
 * (`/ws/documents/{id}`) and falls back to polling `GET .../progress` every
 * 2s if the socket can't connect or drops — the UI never gets stuck with no
 * progress signal just because a proxy/browser blocked the WS.
 *
 * The fallback only ever runs while this effect is live and the job hasn't
 * reached a terminal state: the socket's `close` event fires asynchronously
 * AFTER cleanup (and after our own close on a terminal event), and starting
 * a poll from there used to leak an interval nobody cleared -- it kept
 * hitting `/progress` every 2s, even after leaving the page, forever if
 * that request 404'd. A 4xx is also final: the document/job isn't there, and
 * asking again every 2s can't change that.
 *
 * A socket that opens but never delivers anything (seen in production: the
 * page list advanced to 282/420 while the bar sat at "Uploading 0%") fires
 * neither `error` nor `close`, so it also counts as down once it has been
 * silent for SOCKET_SILENCE_MS: polling covers until the next frame arrives.
 */
export function useDocumentProgress(documentId: string | undefined, enabled: boolean) {
  const [event, setEvent] = useState<ProgressEvent | null>(null);

  useEffect(() => {
    if (!documentId || !enabled) return;
    let cancelled = false;
    let finished = false;
    let socket: WebSocket | null = null;
    let poll: ReturnType<typeof setInterval> | null = null;
    let silence: ReturnType<typeof setTimeout> | null = null;

    const clearSilenceTimer = () => {
      if (silence) {
        clearTimeout(silence);
        silence = null;
      }
    };

    const stopPolling = () => {
      if (poll) {
        clearInterval(poll);
        poll = null;
      }
    };

    const startPolling = () => {
      if (poll || cancelled || finished) return;
      poll = setInterval(async () => {
        try {
          const job = await documentsApi.progress(documentId);
          if (cancelled) return;
          setEvent({
            document_id: documentId,
            job_id: job.id,
            stage: job.stage,
            percent: job.progress_percent,
            page: null,
            message: job.error_message || "",
            status: job.status,
          });
          if (TERMINAL_STATUSES.has(job.status)) {
            finished = true;
            stopPolling();
            clearSilenceTimer();
          }
        } catch (err) {
          const status = axios.isAxiosError(err) ? err.response?.status : undefined;
          if (status !== undefined && status >= 400 && status < 500) {
            finished = true;
            stopPolling();
          }
          // Otherwise (network error / 5xx): transient, keep polling.
        }
      }, 2000);
    };

    const armSilenceTimer = () => {
      clearSilenceTimer();
      if (!cancelled && !finished) silence = setTimeout(startPolling, SOCKET_SILENCE_MS);
    };

    try {
      socket = new WebSocket(wsUrlForDocument(documentId));
      armSilenceTimer();
      socket.onmessage = (msg) => {
        if (cancelled) return;
        try {
          const parsed = JSON.parse(msg.data) as ProgressEvent;
          setEvent(parsed);
          // The socket is delivering again: it's the live source, not polling.
          stopPolling();
          if (TERMINAL_STATUSES.has(parsed.status)) {
            finished = true;
            clearSilenceTimer();
            socket?.close();
          } else {
            armSilenceTimer();
          }
        } catch {
          /* ignore malformed frame */
        }
      };
      socket.onerror = () => startPolling();
      socket.onclose = () => startPolling();
    } catch {
      startPolling();
    }

    return () => {
      cancelled = true;
      clearSilenceTimer();
      socket?.close();
      stopPolling();
    };
  }, [documentId, enabled]);

  return event;
}
