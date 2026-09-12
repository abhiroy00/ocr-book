import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    host: true,
    port: 5173,
    // Running inside Docker on a Windows host: filesystem change events
    // from a bind-mounted volume don't reliably reach the Linux
    // container's file watcher (a well-known Docker Desktop/WSL2 +
    // Windows-host limitation), so Vite never notices a saved file and
    // keeps serving the old module forever -- no error, no HMR, nothing.
    // Polling instead of relying on native fs events fixes this
    // permanently, at the cost of a small, fixed CPU check interval.
    watch: {
      usePolling: true,
      interval: 300,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
  },
});
