import { ConfigEnv, UserConfigExport } from "vite";
import vue from "@vitejs/plugin-vue";
import legacy from "@vitejs/plugin-legacy";
import viteCompression from "vite-plugin-compression";
import { fileURLToPath, URL } from "url";

export default ({ mode }: ConfigEnv): UserConfigExport => {
  const isProd = mode === "production";
  const plugins = [vue(), legacy()];
  if (isProd) {
    plugins.push(viteCompression());
  }
  return {
    plugins,
    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", import.meta.url)),
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            // 將 node_modules 依套件名拆小包
            if (id.includes("node_modules")) {
              const libName = id.split("node_modules/")[1].split("/")[0];
              return libName;
            }
          },
        },
      },
    },
    server: {
      host: "127.0.0.1",
      port: 3000,
    },
  };
};
