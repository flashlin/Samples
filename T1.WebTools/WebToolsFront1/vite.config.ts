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
  base: "/my_path/", // 設定打包後的檔案路徑
  build: {
    assetsDir: "assets", // 設定放置打包後 js/css 的目錄
  }, //最後 `my_path/assets/`
});
