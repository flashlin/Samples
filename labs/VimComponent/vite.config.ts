import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    lib: {
      entry: 'src/vim-editor.ts',
      name: 'VimEditor',
      fileName: (format) => `vim-editor.${format}.js`,
      formats: ['es', 'umd']
    },
    rollupOptions: {
      external: ['p5'],
      output: {
        globals: {
          p5: 'p5'
        }
      }
    }
  },
  server: {
    open: true
  }
}); 