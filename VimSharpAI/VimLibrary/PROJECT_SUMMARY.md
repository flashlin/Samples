# VimLibrary 專案總結

## 專案概述

VimLibrary 是一個使用 TypeScript 開發的 Web Component Library，專門為 VimFront 專案提供 Vim 編輯器元件。

## 技術棧

### 核心技術
- **TypeScript 5.9.3** - 提供型別安全
- **Vite 7.1.7** - 現代化的建置工具
- **pnpm** - 快速且節省空間的套件管理器
- **Web Components** - 原生瀏覽器 API，無框架依賴

### 開發工具
- **vite-plugin-dts 4.5.4** - 自動生成 TypeScript 型別定義
- **Node.js 22.16.0** - 開發環境

## 為什麼選擇 Vite？

✅ **簡單易用** - 配置檔案簡潔，內建 Library Mode  
✅ **現代化** - 使用原生 ESM，開發體驗極佳  
✅ **快速** - HMR 熱更新速度快  
✅ **TypeScript 支援** - 原生支援，無需額外配置  
✅ **輸出格式** - 同時支援 ES Module 和 UMD 格式  

相比之下：
- **Webpack** - 配置複雜，適合大型應用而非函式庫
- **Rollup** - Vite 底層就是用 Rollup，但 Vite 提供更好的開發體驗

## 專案結構

```
VimLibrary/
├── src/
│   ├── components/
│   │   └── VimEditor.ts       # 主要編輯器元件
│   ├── utils/
│   │   └── types.ts           # TypeScript 型別定義
│   └── index.ts               # 入口檔案
├── dist/                      # 建置輸出 (自動生成)
│   ├── vim-library.es.js      # ES Module 格式
│   ├── vim-library.umd.js     # UMD 格式
│   └── index.d.ts             # TypeScript 型別定義
├── index.html                 # Demo 頁面
├── vite.config.ts             # Vite 配置
├── tsconfig.json              # TypeScript 配置
├── package.json               # 專案配置
├── .nvmrc                     # Node 版本鎖定
├── .gitignore
├── README.md                  # 使用文件
├── USAGE.md                   # 整合指南
└── PROJECT_SUMMARY.md         # 本文件
```

## 建置輸出

每次執行 `pnpm build` 會產生：

1. **vim-library.es.js** (~1.4 KB)
   - ES Module 格式
   - 用於現代打包工具 (Vite, Webpack, Rollup)
   - Tree-shaking 友好

2. **vim-library.umd.js** (~1.4 KB)
   - UMD (Universal Module Definition) 格式
   - 可在瀏覽器直接使用 `<script>` 載入
   - 相容於 AMD, CommonJS, 全域變數

3. **index.d.ts**
   - TypeScript 型別定義檔
   - 提供完整的型別提示和自動完成
   - 包含所有 interface 和 class 定義

## 可用指令

```bash
# 開發模式 (啟動開發伺服器)
pnpm dev

# 建置 library
pnpm build

# 型別檢查 (不輸出檔案)
pnpm type-check

# 預覽建置結果
pnpm preview

# 清除建置檔案
pnpm clean

# 清除後重新建置
pnpm rebuild
```

## Package.json 配置重點

```json
{
  "type": "module",                    // 使用 ES Module
  "main": "./dist/vim-library.umd.js", // CommonJS 入口
  "module": "./dist/vim-library.es.js",// ES Module 入口
  "types": "./dist/index.d.ts",        // TypeScript 定義
  "exports": {                         // 現代化的導出定義
    ".": {
      "types": "./dist/index.d.ts",    // 型別定義優先
      "import": "./dist/vim-library.es.js",
      "require": "./dist/vim-library.umd.js"
    }
  },
  "files": ["dist"]                    // 發佈時只包含 dist 目錄
}
```

## 在 VimFront 專案中使用

### 安裝
```bash
# 開發階段使用本地路徑
pnpm add file:../VimLibrary

# 或使用 workspace (推薦)
# 在根目錄建立 pnpm-workspace.yaml
```

### 使用範例
```typescript
import { VimEditor } from 'vim-library'

const editor = new VimEditor()
editor.init({ initialContent: 'Hello!' })
document.body.appendChild(editor)
```

詳見 `USAGE.md` 檔案。

## Web Components 的優勢

1. **無框架依賴** - 可在 React, Vue, Angular, 原生 JavaScript 中使用
2. **樣式隔離** - Shadow DOM 提供樣式封裝
3. **原生支援** - 瀏覽器原生 API，無需額外 polyfill (現代瀏覽器)
4. **可重用性** - 真正的組件化，像使用 HTML 標籤一樣簡單

## 型別定義

Library 提供完整的 TypeScript 支援：

```typescript
interface VimEditorOptions {
  initialContent?: string
  config?: VimConfig
}

interface VimConfig {
  mode?: 'normal' | 'insert' | 'visual'
  readOnly?: boolean
  lineNumbers?: boolean
}

class VimEditor extends HTMLElement {
  init(options?: VimEditorOptions): void
  getContent(): string
  setContent(content: string): void
}
```

## 開發注意事項

1. **Node 版本**: 使用 `.nvmrc` 鎖定版本為 22.16.0
   ```bash
   nvm use
   ```

2. **建置前執行**: 確保先執行 `pnpm install`

3. **型別檢查**: 提交前執行 `pnpm type-check` 確保無型別錯誤

4. **清除建置**: 遇到奇怪問題時執行 `pnpm rebuild`

## 未來擴展建議

1. **測試框架** - 加入 Vitest 進行單元測試
2. **更多元件** - 擴展更多 Vim 相關元件
3. **主題系統** - 支援可自訂的配色方案
4. **鍵盤映射** - 實現完整的 Vim 鍵盤快捷鍵
5. **文件網站** - 使用 VitePress 建立文件網站
6. **Storybook** - 加入元件展示和互動式文件

## 效能指標

- **打包大小**: ~1.4 KB (gzip 後 ~0.6 KB)
- **建置時間**: ~1 秒
- **TypeScript 編譯**: 即時

## 相依套件

### 生產依賴
無 (這是一個零依賴的 library！)

### 開發依賴
- @types/node: TypeScript Node.js 型別定義
- typescript: TypeScript 編譯器
- vite: 建置工具
- vite-plugin-dts: 生成 .d.ts 檔案

## 授權

ISC License

---

**建立時間**: 2025-10-01  
**Node 版本**: 22.16.0  
**套件管理器**: pnpm 10.13.1

