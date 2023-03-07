import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

import { quasar, transformAssetUrls } from "@quasar/vite-plugin";

// https://vitejs.dev/config/
export default defineConfig({
  mode: "development",
  plugins: [
    vue({ template: { transformAssetUrls } }),
    quasar({
      sassVariables: "src/quasar-variables.sass",
    }),
  ],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  base: "/js/LocalQuery/", // 設定打包後的檔案路徑
  build: {
    assetsDir: "assets", // 設定放置打包後 js/css 的目錄, 最後 `my_path/assets/`
    rollupOptions: {
      external: ["node-sql-parser"],
      output: {
        globals: {
          "node-sql-parser": "NodeSqlParser",
        },
        manualChunks(id) {
          if (id.includes("node_modules")) {
            return "vendor";
          }
        },
      },
    },
  },
  define: {
    __VUE_PROD_DEVTOOLS__: true,
    global: "window",
    "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV),
  },
});
