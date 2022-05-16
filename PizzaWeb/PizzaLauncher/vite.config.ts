import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import vueJsx from "@vitejs/plugin-vue-jsx";
import path from "path";
import glob from "glob";
//import { loadEnv } from "vite";
//loadEnv(mode, process.cwd()).VITE_APP_OUT_DIR

export default ({ mode }) => {
	require("dotenv").config({ path: `./.env` });
	require("dotenv").config({ path: `./.env.${mode}` });
	// now you can access config with process.env.{configName}


	return defineConfig({
		plugins: [vue(), vueJsx({})],
		esbuild: {
			jsxFactory: "h",
			jsxFragment: "Fragment",
		},
		resolve: {
			alias: {
				"@": path.resolve(__dirname, "./src"),
			},
		},
		base: mode == "development" ? "" : "dist",
		build: {
			manifest: true,
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
					target: "http://localhost:5129/",
					changeOrigin: true,
					ws: true,
					//rewrite: (pathStr) => pathStr.replace("/api", ""),
				},
				"/images": {
					target: "http://localhost:5129/",
					changeOrigin: true,
					ws: true,
					//rewrite: (pathStr) => pathStr.replace("/api", ""),
				},
			},
		},
	});
};
