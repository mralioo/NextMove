import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: `npm run dev` (http://localhost:5173) proxies /api to the operator API; build: served by the operator API at /app
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  server: { port: 5173, proxy: { "/api": { target: process.env.OPERATOR_API || "http://127.0.0.1:8770", changeOrigin: true, timeout: 300000 } } },
});
