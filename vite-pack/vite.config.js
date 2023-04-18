// vite.config.js
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.ts",
      name: "index",
      fileName: (format) => `index.${format}.js`,
    },
    rollupOptions: {
      // Make sure to externalize deps that shouldn't be bundled into your library
      external: ["react"],
      output: {
        // Provide global variables to use in the UMD build
        // for externalized deps
        globals: {
          react: "React",
        },
      },
      input: {
        main: "./index.html",
      },
    },
  },
});
