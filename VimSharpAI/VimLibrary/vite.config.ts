import { defineConfig } from 'vite'
import { resolve } from 'path'
import dts from 'vite-plugin-dts'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    dts({
      insertTypesEntry: true,
      rollupTypes: true
    })
  ],
  build: {
    lib: {
      // Entry point for the library
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'VimLibrary',
      // Output file names
      fileName: (format) => `vim-library.${format}.js`
    },
    rollupOptions: {
      // Externalize deps that shouldn't be bundled
      external: [],
      output: {
        // Global variables to use in the UMD build
        globals: {}
      }
    }
  }
})

