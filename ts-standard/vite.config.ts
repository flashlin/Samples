import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'dist',
    lib: {
      entry: 'src/index.ts',
      name: 'MyLibrary',
      fileName: 'my-library',
    },
    rollupOptions: {
      // 打包過程中排除不必要的依賴
      external: ['lodash'],
      output: {
        // 將庫打包為 UMD 格式
        format: 'umd',
        // 在 UMD 模式下，指定全局變數的名稱
        globals: {
          lodash: '_',
        },
      },
    },
  },
  optimizeDeps: {
    include: ['axios', 'lodash'],
  },
});
