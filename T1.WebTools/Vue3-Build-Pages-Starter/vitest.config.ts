/// <reference types="vitest" />
import { defineConfig } from 'vite'
import Vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from "url"
import VueI18nPlugin from '@intlify/unplugin-vue-i18n/vite'
import { resolve } from 'path'

export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  plugins: [
    Vue(),
    VueI18nPlugin({
      include: resolve(__dirname, './**/locales/**'),
    }),
  ],
  test: {
    setupFiles: ['./vitest.setup.ts'],
    globals: true,
    environment: 'jsdom',
  },
})
