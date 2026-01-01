import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'
import { fileURLToPath } from 'url'
import { dirname } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

export default defineConfig({
  root: 'demo',
  base: '/Samples/T1WebComponents/t1-web-components/',
  plugins: [vue()],
  build: {
    outDir: '../dist-demo',
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@lib': resolve(__dirname, 'src/lib'),
      '@': resolve(__dirname, 'demo')
    }
  },
  server: {
    open: true
  }
})
