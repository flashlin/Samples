import { defineConfig } from "vite";
import replace from "@rollup/plugin-replace";
import { resolve } from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  build: {
    rollupOptions: {
      plugins: [
        //  Toggle the booleans here to enable / disable Phaser 3 features:
        replace({
          "typeof CANVAS_RENDERER": "'true'",
          "typeof WEBGL_RENDERER": "'true'",
          "typeof EXPERIMENTAL": "'true'",
          "typeof PLUGIN_CAMERA3D": "'false'",
          "typeof PLUGIN_FBINSTANT": "'false'",
          "typeof FEATURE_SOUND": "'true'",
        }),
      ],
    },
  },
});
