import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'dist',
    lib: {
      entry: 'src/index.ts',
      name: 'ts-standard',
      fileName: 'ts-standard',
    },
    rollupOptions: {
      // 打包過程中排除不必要的依賴
      external: ['lodash'],
      output: [
        {
          format: 'umd',
          name: 'ts-standard-umd',
          globals: {
            lodash: '_',
          },
        },
        {
          format: 'es',
          sourcemap: true,
          exports: 'named',
          chunkFileNames: 'chunks/[name]-[hash].js',
        },
        {
          format: 'es',
          sourcemap: true,
          exports: 'named',
          chunkFileNames: 'chunks/[name]-[hash].mjs',
        },
      ]
    },
  },
  optimizeDeps: {
    include: ['axios', 'lodash'],
  },
});
