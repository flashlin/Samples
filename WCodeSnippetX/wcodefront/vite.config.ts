import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'url';

export default defineConfig({
  plugins: [
    vue(),
  ],
  build: {
    outDir: "../WCodeSnippetX/bin/Debug/net6.0-windows/views",
    emptyOutDir: true,
    manifest: true,
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL('./src', import.meta.url)),
    }
  }
})
