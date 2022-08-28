import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    rollupOptions: {
      output: {
        //chunkFileNames: "assets/js/[name]-[hash].js",
        entryFileNames: "assets/[name].js",
        //assetFileNames: "assets/[ext]/[name]-[hash].[ext]",
        assetFileNames: "assets/[name].[ext]",
      },
    },
  },
});
