import typescript from "rollup-plugin-typescript2";
import serve from "rollup-plugin-serve";
import livereload from "rollup-plugin-livereload";
import image from '@rollup/plugin-image';
import { string as text } from 'rollup-plugin-string';

export default {
  input: "src/f4.ts",
  output: {
    file: "public/bundle.js",
    format: "iife",
  },
  plugins: [
    typescript(),
    serve({
      open: true,
      contentBase: "public",
      port: 5001,
    }),
    livereload(),
    image(),
    text({
      include: 'assets/**/*.txt',
    }),
  ],
};
