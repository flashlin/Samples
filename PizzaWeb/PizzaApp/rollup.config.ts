import typescript from "@rollup/plugin-typescript";
import { nodeResolve } from '@rollup/plugin-node-resolve';
import json from '@rollup/plugin-json';
import { terser } from 'rollup-plugin-terser'

export default {
  input: "src/pages/luncher.ts",
  output: {
    dir: "../PizzaWeb/wwwroot/dist/pages",
    format: "esm",  //iife, cjs, esm
    globals: {
      'jquery': '$'
    }
  },
  plugins: [
    typescript(),
    json(),
    nodeResolve(),
    // 如果要在瀏覽器使用需要加入設定如下
    //nodeResolve({ browser: true, preferBuiltins: true }),
    //terser()
  ],
};
