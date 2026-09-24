import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: `make ui` (http://localhost:3000, hot reload) proxies /api to the operator API; build: served by the operator API at /app/ (make up)
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  server: { port: 3000, proxy: { "/api": { target: process.env.OPERATOR_API || "http://127.0.0.1:8770", changeOrigin: true, timeout: 300000 } } },
});
