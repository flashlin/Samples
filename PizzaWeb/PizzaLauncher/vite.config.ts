import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import vueJsx from "@vitejs/plugin-vue-jsx";
import path from "path";
import glob from "glob";

export default defineConfig({
	plugins: [
		vue(),
		vueJsx({}),
	],
	esbuild: {
		jsxFactory: "h",
		jsxFragment: "Fragment",
	},
	resolve: {
		alias: {
			"@": path.resolve(__dirname, "./src"),
		},
	},
	base: "dist",
	build: {
		outDir: path.resolve(__dirname, "../PizzaWeb/wwwroot/dist"),
		assetsDir: "assets",
		rollupOptions: {
			input: glob.sync(path.resolve(__dirname, "spa", "*.html")),
		},
	},
	server: {
		host: "localhost",
		port: 3000,
		open: true,
		https: false,
		proxy: {
			"/api": {
				target: "http://localhost:3333/",
				changeOrigin: true,
				ws: true,
				rewrite: (pathStr) => pathStr.replace("/api", ""),
			},
		},
	},
});
