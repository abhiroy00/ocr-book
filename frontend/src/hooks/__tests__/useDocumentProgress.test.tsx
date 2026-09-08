import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useDocumentProgress } from "@/hooks/useDocumentProgress";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
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
    this.onclose?.();
  }
  emit(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

describe("useDocumentProgress", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    // @ts-expect-error - test double
    global.WebSocket = MockWebSocket;
  });
  afterEach(() => {
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
});
