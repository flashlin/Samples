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
        match: ['*://*.coreop.net/*', 'https://www.w3schools.com/*'],
      },
      build: {
        externalGlobals: {
          vue: cdn.jsdelivr('Vue', 'dist/vue.global.prod.js'),
        },
      },
    }),
  ],
  resolve: {
    alias: {
      '@': '/src/'
    }
  },
  // build: {
  //   rollupOptions: {
  //     external: [
  //       'monaco-editor/esm/vs/editor/editor.worker',
  //       'monaco-editor/esm/vs/language/json/json.worker',
  //       'monaco-editor/esm/vs/language/css/css.worker',
  //       'monaco-editor/esm/vs/language/html/html.worker',
  //       'monaco-editor/esm/vs/language/typescript/ts.worker',
  //       //'monaco-editor/esm/vs/editor/contrib/find/findController'
  //     ],
  //   },
  // },
  optimizeDeps: {
    exclude: ['sql-wasm'],
    //esbuildOptions: { target: 'es2020' },
    // include: [
    //   'monaco-editor/esm/vs/editor/editor.worker',
    //   'monaco-editor/esm/vs/language/json/json.worker',
    //   'monaco-editor/esm/vs/language/css/css.worker',
    //   'monaco-editor/esm/vs/language/html/html.worker',
    //   'monaco-editor/esm/vs/language/typescript/ts.worker',
    // ],
  },
});
