# Vim Editor Demo

這是一個使用 Vue 3 + TypeScript 的示範專案，展示如何整合和使用 VimComponent。

## 功能特色

- ✨ 使用 VimComponent 的編輯器（寬度 90%，高度 300px）
- 🎯 即時程式碼執行
- 📊 執行結果表格顯示
- 💚 Vue 3 + TypeScript + Vite

## 專案結構

```
VimDemo/
├── src/
│   ├── views/
│   │   └── App.vue          # 主應用程式元件
│   ├── main.ts              # 應用程式入口
│   ├── style.css            # 全域樣式
│   └── vite-env.d.ts        # TypeScript 型別定義
├── node_modules/
│   └── vimcomponent/        # 本地安裝的 VimComponent package
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## 安裝與執行

### 安裝依賴

此專案使用本地的 VimComponent package，透過 pnpm 的 file: 協議安裝：

```bash
pnpm install
```

> **注意**：
> - VimComponent 是從 `../VimComponent` 本地路徑安裝的（在 package.json 中定義為 `"vimcomponent": "file:../VimComponent"`）
> - VimComponent 需要 **peer dependencies**：`lit` 和 `p5@^1.6.0`，這些已自動安裝在 dependencies 中
> - **重要**：必須使用 p5.js 1.x 版本（如 1.11.10），不要使用 2.x 版本

### 啟動開發伺服器

```bash
pnpm run dev
```

然後在瀏覽器中開啟顯示的 URL（通常是 `http://localhost:5173`）。

### 建置生產版本

```bash
pnpm run build
```

### 預覽生產版本

```bash
pnpm run preview
```

## 使用說明

1. 在 Vim 編輯器中輸入或編輯 JavaScript/TypeScript 程式碼
2. 點擊 "Run" 按鈕執行程式碼
3. 執行結果會顯示在下方的表格中，包含：
   - 時間戳記
   - 程式碼內容（前 100 字元）
   - 執行結果或錯誤訊息

> **重要**：VimComponent 的 `content` 屬性需要 **字串陣列（`string[]`）**，每個元素代表一行。
> 詳細用法請參考 [USAGE.md](./USAGE.md)。

## 技術細節

- **編輯器配置**：
  - 寬度：90%
  - 高度：300px
  - 支援 Vim 快捷鍵和模式

- **結果表格**：
  - 顯示執行歷史
  - 成功執行顯示為綠色
  - 錯誤顯示為紅色
  - 支援多行輸出

## 依賴套件

### 主要依賴
- Vue 3.5.22
- VimComponent 1.0.0（本地 package）
- lit 3.3.1（VimComponent 的 peer dependency）
- p5 1.11.10（VimComponent 的 peer dependency，必須使用 1.x 版本）

### 開發依賴
- Vite 7.1.9
- TypeScript 5.9.3
- @vitejs/plugin-vue 6.0.1
- vue-tsc 3.1.1

### Peer Dependencies 說明

VimComponent 使用 `peerDependencies` 來宣告它需要 `lit` 和 `p5` 套件。這表示：
- ✅ 使用專案需要自行安裝這些依賴
- ✅ 避免重複安裝相同的套件
- ✅ 讓使用專案控制版本（在相容範圍內）
- ⚠️ p5.js 必須使用 1.x 版本（^1.6.0），目前不支援 2.x 版本

## 授權

ISC

