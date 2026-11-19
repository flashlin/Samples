import { build } from "esbuild";
import { cp, rm } from "node:fs/promises";
import path from "node:path";

const outdir = "dist";

const entries = [
  { entry: "src/background/index.ts", out: "background", format: "esm" },
  { entry: "src/content/index.ts", out: "content", format: "iife" },
  { entry: "src/offscreen/index.ts", out: "offscreen", format: "esm" },
  { entry: "src/popup/index.ts", out: "popup", format: "esm" },
  { entry: "src/options/index.ts", out: "options", format: "esm" },
];

await rm(outdir, { recursive: true, force: true });
await cp("static", outdir, { recursive: true });
await Promise.all(
  entries.map(({ entry, out, format }) =>
    build({ entryPoints: [entry], outfile: `${outdir}/${out}.js`, bundle: true, format, target: "chrome116", minify: true, logLevel: "warning" }),
  ),
);
console.log(`Built extension into ${path.resolve(outdir)}`);
console.log("Load it from chrome://extensions → Developer mode → Load unpacked");
