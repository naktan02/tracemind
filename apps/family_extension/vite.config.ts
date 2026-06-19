import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

function contentScriptBundleGuard(): Plugin {
  return {
    name: "tracemind-content-script-bundle-guard",
    generateBundle(_options, bundle) {
      const contentChunk = bundle["assets/content.js"];
      if (contentChunk?.type !== "chunk") {
        this.error("content script bundle was not emitted at assets/content.js");
        return;
      }
      if (
        contentChunk.imports.length > 0 ||
        contentChunk.dynamicImports.length > 0
      ) {
        this.error(
          "content script must be emitted as a standalone classic script",
        );
      }
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [react(), contentScriptBundleGuard()],
  build: {
    rollupOptions: {
      input: {
        popup: "popup.html",
        child: "index.html",
        parent: "parent.html",
        collectorDebug: "collector-debug.html",
        content: "src/collector/content.ts",
        background: "src/extension/background.ts",
      },
      output: {
        entryFileNames: "assets/[name].js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
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
