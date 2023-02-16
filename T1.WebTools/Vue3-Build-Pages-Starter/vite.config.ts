import { fileURLToPath, URL } from 'url'
import type { ConfigEnv, UserConfigExport } from 'vite'
import { loadEnv } from 'vite'
import mpa from "vite-plugin-mpa"
import vue from './build/customVuePlugin'
import legacy from "@vitejs/plugin-legacy"
import VueI18nPlugin from '@intlify/unplugin-vue-i18n/vite'
import { resolve } from "path"
import importToCDN from 'vite-plugin-cdn-import'

export default ( { mode }: ConfigEnv ): UserConfigExport => {
  const viteEnv = loadEnv(mode, process.cwd())
  const plugins = [
    vue(),
    mpa(),
    legacy(),
    VueI18nPlugin({
      include: resolve(__dirname, './**/locales/**'),
    }),
  ]
  if (mode !== 'production') {
    const myCDNPathDict: Record<string, string> = {
      staging: 'https://img-1.cdnnetworks.net',
      uat: 'https://img-2.cdnnetworks.net',
    }
    plugins.push(
      importToCDN({
        prodUrl: `${ myCDNPathDict[mode] }/js/{path}`,
        modules: [
          {
            name: 'my-auth',
            var: 'createLoginClient',
            path: 'latest/index.min.js',
          },
        ],
      }),
    )
  }

  return {
    base: viteEnv.VITE_CDN_URL,
    plugins,
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    build: {
      manifest: true,
    },
  }
}
