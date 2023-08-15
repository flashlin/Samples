import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// https://vitejs.dev/config/
export default defineConfig({
  base: './',
  plugins: [
    vue(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    target: ['es2020'],
    manifest: true,
    rollupOptions: {
      output: {
        assetFileNames: 'assets/[name]-[hash][extname]',
        entryFileNames: '[name].[hash].js',
      },
    },
  },
  optimizeDeps: {
    //exclude: ['sql-wasm'],
    esbuildOptions: { target: 'es2020' },
  },
  server: {
    port: 8001,
    open: true,
  },
});
