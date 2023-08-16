import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import monkey, { cdn } from 'vite-plugin-monkey';
import dts from 'vite-plugin-dts';

export default defineConfig({
  base: './',
  plugins: [
    vue(),
    dts({
      tsconfigPath: 'tsconfig.monkey.json',
    }),
    monkey({
      entry: 'src/main.ts',
      userscript: {
        icon: 'https//vitejs.dev/logo.svg',
        namespace: 'flash-knife',
        match: ['*://*/*'],
      },
      build: {
        externalGlobals: {
          vue: cdn.jsdelivr('Vue', 'dist/vue.global.prod.js'),
        },
      },
    }),
  ],
  // resolve: {
  //   alias: {
  //     '@': './',
  //   },
  // },
  // optimizeDeps: {
  //   //exclude: ['sql-wasm'],
  //   esbuildOptions: { target: 'es2020' },
  //},
});
