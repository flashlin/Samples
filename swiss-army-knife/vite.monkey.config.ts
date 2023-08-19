import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import monkey, { cdn } from 'vite-plugin-monkey';
import dts from 'vite-plugin-dts';
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  base: './',
  plugins: [
    AutoImport({
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      resolvers: [ElementPlusResolver()],
    }),
    vue(),
    dts({
      tsconfigPath: 'tsconfig.monkey.json',
    }),
    monkey({
      entry: 'src/main.ts',
      userscript: {
        version: '0.1',
        icon: 'https//vitejs.dev/logo.svg',
        namespace: 'flash-knife',
        match: ['*://dba-*.coreop.net/*', 'https://www.w3schools.com/*'],
      },
      build: {
        externalGlobals: {
          vue: cdn.jsdelivr('Vue', 'dist/vue.global.prod.js'),
        },
      },
    }),
  ],
  optimizeDeps: {
    exclude: ['sql-wasm'],
    //esbuildOptions: { target: 'es2020' },
  },
});
