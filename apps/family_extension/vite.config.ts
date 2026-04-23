import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        popup: "index.html",
        parent: "parent.html",
      },
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5174,
  },
  preview: {
    host: "0.0.0.0",
    port: 4174,
  },
});
