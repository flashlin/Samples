import typescript from "@rollup/plugin-typescript";

export default {
  input: "src/pages/index.ts",
  output: {
    dir: "../PizzaWeb/wwwroot/dist/pages",
    format: "esm",
  },
  plugins: [
    typescript(),
  ],
};
