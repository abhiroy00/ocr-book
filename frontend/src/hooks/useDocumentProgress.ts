import { useEffect, useState } from "react";
import axios from "axios";
import { documentsApi, wsUrlForDocument } from "@/services/api";
import type { ProgressEvent } from "@/types/document";

// Missing "CANCELLED" here was a real bug: a stopped job never closed its
// WebSocket / never stopped its polling fallback, both left running for
// the rest of the component's lifetime.
const TERMINAL_STATUSES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

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
 */
export function useDocumentProgress(documentId: string | undefined, enabled: boolean) {
  const [event, setEvent] = useState<ProgressEvent | null>(null);

  useEffect(() => {
    if (!documentId || !enabled) return;
    let cancelled = false;
    let finished = false;
    let socket: WebSocket | null = null;
    let poll: ReturnType<typeof setInterval> | null = null;

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

    try {
      socket = new WebSocket(wsUrlForDocument(documentId));
      socket.onmessage = (msg) => {
        if (cancelled) return;
        try {
          const parsed = JSON.parse(msg.data) as ProgressEvent;
          setEvent(parsed);
          if (TERMINAL_STATUSES.has(parsed.status)) {
            finished = true;
            socket?.close();
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
      socket?.close();
      stopPolling();
    };
  }, [documentId, enabled]);

  return event;
}
