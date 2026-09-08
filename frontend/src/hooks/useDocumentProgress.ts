import { useEffect, useRef, useState } from "react";
import { documentsApi, wsUrlForDocument } from "@/services/api";
import type { ProgressEvent } from "@/types/document";

const TERMINAL_STATUSES = new Set(["COMPLETED", "FAILED"]);

/**
 * Live processing progress (spec section 20): prefers the WebSocket
 * (`/ws/documents/{id}`) and falls back to polling `GET .../progress` every
 * 2s if the socket can't connect or drops — the UI never gets stuck with no
 * progress signal just because a proxy/browser blocked the WS.
 */
export function useDocumentProgress(documentId: string | undefined, enabled: boolean) {
  const [event, setEvent] = useState<ProgressEvent | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!documentId || !enabled) return;
    let cancelled = false;
    let socket: WebSocket | null = null;

    const startPolling = () => {
      if (pollRef.current) return;
      pollRef.current = setInterval(async () => {
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
          if (TERMINAL_STATUSES.has(job.status) && pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
        } catch {
          // Backend not reachable yet (job may not exist); keep polling.
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
          if (TERMINAL_STATUSES.has(parsed.status)) socket?.close();
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
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [documentId, enabled]);

  return event;
}
