import type { ConfigEnv, UserConfigExport } from 'vite';
import vue from "./build/customVuePlugin";
import vueJsx from "@vitejs/plugin-vue-jsx";
import legacy from '@vitejs/plugin-legacy';
import viteCompression from 'vite-plugin-compression';
import { fileURLToPath, URL } from "url";

export default ({ mode }: ConfigEnv): UserConfigExport => {
  const isProd = mode === 'production';
  const plugins = [
    vue(),
    vueJsx(),
    legacy(),
  ];
  if (isProd) {
    plugins.push(viteCompression());
  }
  return {
    plugins,
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    server: {
      host: 'sqlite.localdev.net',
      port: 3001,
    },
  };
};