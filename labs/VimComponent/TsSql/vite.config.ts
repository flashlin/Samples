import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    lib: {
      entry: 'src/index.ts',
      name: 'TsSql',
      fileName: (format) => `tssql.${format}.js`,
      formats: ['es', 'umd']
    }
  },
  server: {
    open: true
  }
});

