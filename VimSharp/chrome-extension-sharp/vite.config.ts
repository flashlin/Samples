import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'
import { writeFileSync, copyFileSync } from 'fs'
import fs from 'fs'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    {
      name: 'copy-popup-html',
      closeBundle() {
        // 複製 popup.html 到 dist 目錄
        copyFileSync('public/popup.html', 'dist/popup.html')
      }
    },
    {
      name: 'wasm-mime',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url && req.url.endsWith('.wasm')) {
            res.setHeader('Content-Type', 'application/wasm');
          }
          next();
        });
      }
    },
    {
      name: 'wa-sqlite-wasm-alias',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url === '/node_modules/.vite/deps/wa-sqlite.wasm') {
            const wasmPath = path.resolve(__dirname, 'public/wa-sqlite.wasm')
            if (fs.existsSync(wasmPath)) {
              res.setHeader('Content-Type', 'application/wasm')
              fs.createReadStream(wasmPath).pipe(res)
              return
            }
          }
          next()
        })
      }
    }
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src')
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    cssCodeSplit: false,
    target: 'esnext',
    sourcemap: true,
    rollupOptions: {
      input: {
        popup: resolve(__dirname, 'src/popup.ts'),
        content: resolve(__dirname, 'src/content.ts'),
        background: resolve(__dirname, 'src/background.ts')
      },
      output: {
        entryFileNames: (chunkInfo) => {
          if (chunkInfo.name === 'content' || chunkInfo.name === 'background') {
            return 'js/[name].js'
          }
          return '[name].js'
        },
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: (assetInfo) => {
          if (assetInfo.name === 'style.css') {
            return 'popup.css'
          }
          return 'assets/[name].[hash].[ext]'
        },
        manualChunks(id) {
          const vendors = ['vue', 'monaco-editor', 'monaco-vim', 'papaparse', 'xlsx', 
            'axios', 'handlebars', 'wa-sqlite', 
            '@univerjs/presets', 
          ];
          for (const vendor of vendors) {
            if (id.includes(`node_modules/${vendor}`)) {
              return `vendor-${vendor}`;
            }
          }
        }
      }
    }
  },
  esbuild: {
    target: 'esnext',
    tsconfigRaw: {
      compilerOptions: {
        experimentalDecorators: true
      }
    }
  },
  server: {
    middlewareMode: false,
  }
})
