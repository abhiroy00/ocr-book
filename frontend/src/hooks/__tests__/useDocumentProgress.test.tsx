import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { AxiosError, AxiosHeaders } from "axios";
import { useDocumentProgress } from "@/hooks/useDocumentProgress";
import { documentsApi } from "@/services/api";
import type { ProcessingJob } from "@/types/document";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  // Real browsers dispatch `close` asynchronously, after the caller's
  // close() returns -- i.e. after an effect cleanup has already finished.
  static asyncClose = false;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }
  send() {}
  close() {
    this.closed = true;
    if (MockWebSocket.asyncClose) setTimeout(() => this.onclose?.(), 0);
    else this.onclose?.();
  }
  emit(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

describe("useDocumentProgress", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    MockWebSocket.asyncClose = false;
    // @ts-expect-error - test double
    global.WebSocket = MockWebSocket;
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("surfaces progress events received over the websocket", async () => {
    const { result } = renderHook(() => useDocumentProgress("doc-1", true));

    await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));
    const socket = MockWebSocket.instances[0];
    expect(socket.url).toContain("doc-1");

    act(() =>
      socket.emit({
        document_id: "doc-1",
        job_id: "job-1",
        stage: "ocr",
        percent: 50,
        page: 3,
        message: "Running OCR",
        status: "OCR_PROCESSING",
      }),
    );

    await waitFor(() => expect(result.current?.percent).toBe(50));
    expect(result.current?.stage).toBe("ocr");
  });

  it("closes the socket once a terminal status arrives", async () => {
    const { result } = renderHook(() => useDocumentProgress("doc-2", true));
    await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));
    const socket = MockWebSocket.instances[0];

    act(() =>
      socket.emit({ document_id: "doc-2", job_id: "job-1", stage: "done", percent: 100, page: null, message: "", status: "COMPLETED" }),
    );

    await waitFor(() => expect(result.current?.status).toBe("COMPLETED"));
    expect(socket.closed).toBe(true);
  });

  it("does nothing when disabled", () => {
    renderHook(() => useDocumentProgress("doc-3", false));
    expect(MockWebSocket.instances.length).toBe(0);
  });

  it("never starts polling from a socket close that lands after unmount", async () => {
    vi.useFakeTimers();
    MockWebSocket.asyncClose = true;
    const progress = vi.spyOn(documentsApi, "progress").mockResolvedValue(job("OCR_PROCESSING"));
    const { unmount } = renderHook(() => useDocumentProgress("doc-4", true));
    unmount();
    await act(() => vi.advanceTimersByTimeAsync(10_000));
    expect(progress).not.toHaveBeenCalled();
  });

  it("does not fall back to polling after the socket closes on a terminal event", async () => {
    vi.useFakeTimers();
    MockWebSocket.asyncClose = true;
    const progress = vi.spyOn(documentsApi, "progress").mockResolvedValue(job("COMPLETED"));
    renderHook(() => useDocumentProgress("doc-5", true));
    act(() =>
      MockWebSocket.instances[0].emit({ document_id: "doc-5", job_id: "job-1", stage: "done", percent: 100, page: null, message: "", status: "COMPLETED" }),
    );
    await act(() => vi.advanceTimersByTimeAsync(10_000));
    expect(progress).not.toHaveBeenCalled();
  });

  it("polls when the socket fails and stops for good on a 404", async () => {
    vi.useFakeTimers();
    const progress = vi.spyOn(documentsApi, "progress").mockRejectedValue(notFound());
    renderHook(() => useDocumentProgress("doc-6", true));
    act(() => MockWebSocket.instances[0].onerror?.());
    await act(() => vi.advanceTimersByTimeAsync(20_000));
    expect(progress).toHaveBeenCalledTimes(1);
  });

  it("keeps polling through transient network errors and stops at a terminal status", async () => {
    vi.useFakeTimers();
    const progress = vi
      .spyOn(documentsApi, "progress")
      .mockRejectedValueOnce(new AxiosError("Network Error"))
      .mockResolvedValueOnce(job("OCR_PROCESSING"))
      .mockResolvedValue(job("FAILED"));
    const { result } = renderHook(() => useDocumentProgress("doc-7", true));
    act(() => MockWebSocket.instances[0].onerror?.());
    await act(() => vi.advanceTimersByTimeAsync(20_000));
    expect(progress).toHaveBeenCalledTimes(3);
    expect(result.current?.status).toBe("FAILED");
  });

  it("falls back to polling when an open socket stays silent, and stops once it delivers again", async () => {
    vi.useFakeTimers();
    const progress = vi.spyOn(documentsApi, "progress").mockResolvedValue(job("OCR_PROCESSING"));
    const { result } = renderHook(() => useDocumentProgress("doc-8", true));

    await act(() => vi.advanceTimersByTimeAsync(9_000));
    expect(progress).not.toHaveBeenCalled();

    await act(() => vi.advanceTimersByTimeAsync(3_000));
    expect(progress).toHaveBeenCalledTimes(1);
    expect(result.current?.percent).toBe(50);

    act(() =>
      MockWebSocket.instances[0].emit({ document_id: "doc-8", job_id: "job-1", stage: "ocr", percent: 70, page: 5, message: "", status: "OCR_PROCESSING" }),
    );
    const callsAfterFrame = progress.mock.calls.length;
    await act(() => vi.advanceTimersByTimeAsync(8_000));
    expect(progress.mock.calls.length).toBe(callsAfterFrame);
    expect(result.current?.percent).toBe(70);
  });
});

function job(status: string): ProcessingJob {
  return {
    id: "job-1",
    document_id: "doc",
    status,
    stage: "ocr",
    progress_percent: 50,
    error_message: null,
  } as unknown as ProcessingJob;
}

function notFound(): AxiosError {
  const headers = new AxiosHeaders();
  return new AxiosError("Request failed with status code 404", "ERR_BAD_REQUEST", undefined, undefined, {
    status: 404,
    statusText: "Not Found",
    data: { detail: "No processing job found for this document" },
    headers,
    config: { headers },
  });
}
