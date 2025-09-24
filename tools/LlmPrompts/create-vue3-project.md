# 前端建置規範

先建立 .nvmrc 檔案
並指定 nodejs 版本為 22.16.0
用 nvm 指令切換為 22.16.0 開始建置

## 1. 建立專案資料夾
```bash
Project=CodeBoyFront
mkdir $Project && cd $Project
```

## 2. 使用 `pnpm` 建立 Vue3 SPA 專案
```bash
pnpm create vite@latest . --template vue-ts
```

## 3. 安裝必要套件

### 基礎依賴
```bash
pnpm add vue-router@4 pinia axios
```

### 開發工具
```bash
pnpm add -D tailwindcss@3 postcss autoprefixer
pnpm add -D eslint prettier eslint-plugin-vue @vue/eslint-config-prettier
pnpm add -D typescript vite
```

### Mock 工具
```bash
pnpm add mockjs
pnpm add -D vite-plugin-mock
```

### 測試工具
```bash
pnpm add -D jest ts-jest @types/jest babel-jest
pnpm add -D @vue/test-utils @vue/vue3-jest vue-jest
```

---

## 4. 初始化 TailwindCSS
```bash
pnpm exec tailwindcss init -p
```

`tailwind.config.js`
```js
module.exports = {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: { extend: {} },
  plugins: [],
};
```

全局樣式 (`src/assets/main.css`)
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

html {
  @apply dark;
}
```

---

## 5. 專案目錄結構
```
CodeBoyFront/
├─ public/
├─ mock/             # mock 服務 (app.js、user.js)
├─ src/
│  ├─ apis/          # axios 請求封裝
│  ├─ components/    # 共用元件
│  ├─ views/         # 頁面
│  ├─ assets/
│  ├─ main.ts
├─ .env.development
├─ .env.production
├─ vite.config.js
├─ jest.config.js
```

---

## 6. Vite 設定
`vite.config.js`
```js
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { viteMockServe } from 'vite-plugin-mock';
import path from 'path';

export default defineConfig(({ command }) => {
  return {
    plugins: [
      vue(),
      viteMockServe({
        mockPath: 'mock',
        localEnabled: command === 'serve',
        prodEnabled: command !== 'serve',
      }),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
  };
});
```

---

## 7. Jest 設定
`jest.config.js`
```js
module.exports = {
  preset: 'ts-jest',
  globals: {},
  testEnvironment: 'jsdom',
  testEnvironmentOptions: {
    customExportConditions: ['node', 'node-addons'],
  },
  transform: {
    '^.+\.vue$': '@vue/vue3-jest',
    '^.+\js$': 'babel-jest',
  },
  moduleFileExtensions: ['vue', 'js', 'json', 'jsx', 'ts', 'tsx', 'node'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
};
```

---

## 8. TypeScript 設定
`tsconfig.json`
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

---

## 9. ESLint + Prettier
`.eslintrc.cjs`
```js
module.exports = {
  root: true,
  env: { browser: true, es2021: true },
  extends: [
    'eslint:recommended',
    'plugin:vue/vue3-recommended',
    '@vue/eslint-config-prettier',
  ],
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module' },
  rules: {},
};
```

`.prettierrc`
```json
{
  "semi": true,
  "singleQuote": true,
  "printWidth": 100
}
```

---

## ✅ 完成後功能
- Vue3 SPA (TypeScript + Vite)
- Tailwind3 暗黑模式
- Vue Router + Pinia + Axios
- Mock.js + vite-plugin-mock
- Jest + Vue Test Utils 單元測試
- ESLint + Prettier 格式化規範
