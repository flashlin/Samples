# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

T1 Web Components 是一個基於 Vue 3 + TypeScript + Tailwind 3 的暗黑模式元件庫，主要特色是支援 Camel Case 搜尋和黃色高亮顯示。這是一個可發布的 npm 套件專案。

## 開發指令

```bash
# 啟動開發伺服器（用於測試 App.vue 中的 demo）
pnpm dev

# 編譯 TypeScript 並建置套件（輸出到 dist/）
pnpm build

# 預覽建置結果
pnpm preview
```

## 架構設計

### 專案結構

```
src/
├── App.vue                      # Demo 展示頁面（非套件的一部分）
├── index.ts                     # 套件入口點，匯出所有元件和工具函式
├── components/
│   ├── AutoComplete.vue         # 自動完成元件（允許自由輸入）
│   ├── DropDownList.vue         # 下拉清單元件（強制選取）
│   ├── JsonEditor.vue           # JSON 編輯器元件（支援 CRUD）
│   └── autoCompleteUtils.ts     # Camel Case 搜尋和高亮工具函式
```

### 核心技術架構

**1. Camel Case 搜尋機制**
- `autoCompleteUtils.ts` 提供核心搜尋邏輯
- 支援駝峰命名法搜尋（例如輸入 `VC` 可匹配 `VueComponent`）
- `normalizeText()`: 將 Camel Case 轉為小寫並移除分隔符
- `splitIntoWords()`: 將 Camel Case 文字分解為詞彙陣列
- `highlightText()`: 返回帶有黃色高亮標籤的 HTML 字串

**2. 元件設計模式**
- **DropDownList**: 強制選擇模式，不允許自由輸入，使用 `v-model` 綁定選中值
- **AutoComplete**: 自由輸入模式，允許任意文字，提供智慧推薦
- **JsonEditor**: 表格編輯模式，支援 schema 驗證和 CRUD 操作

兩個搜尋元件都共享相同的搜尋邏輯（`autoCompleteUtils.ts`），但行為不同：
- DropDownList 會在輸入不符合選項時還原為空值
- AutoComplete 允許保留任意輸入值

**3. Schema 定義約定**

JsonEditor 使用 schema 定義資料結構：

```typescript
interface JsonSchemaField {
  key: string                          // 欄位鍵名
  label?: string                       // 顯示標籤（可選）
  type: 'string' | 'number' | 'date'  // 欄位類型
}
```

在 App.vue 或使用元件時，必須使用 `as const` 來確保類型正確推斷：

```typescript
const schema = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Name', type: 'string' as const }
]
```

**4. 套件建置配置**

- Vite 使用 library mode 建置 UMD 和 ES 模組
- `vite-plugin-dts` 自動生成 TypeScript 宣告檔
- Vue 被標記為 `external`，作為 peer dependency
- 建置輸出：
  - `dist/t1-web-components.es.js` - ES 模組
  - `dist/t1-web-components.umd.js` - UMD 模組
  - `dist/index.d.ts` - TypeScript 型別定義

### Demo 頁面設計規範（App.vue）

**配色約定**：
- DropDownList: 藍色系（`bg-blue-500`, `text-blue-400`）
- AutoComplete: 綠色系（`bg-emerald-500`, `text-emerald-400`）
- JsonEditor (Array): 紫色系（`bg-purple-500`, `text-purple-400`）
- JsonEditor (Object): 橙色系（`bg-orange-500`, `text-orange-400`）

**卡片結構模式**：
```vue
<section class="p-6 bg-gray-800/50 rounded-xl border border-gray-700 shadow-lg">
  <h2 class="text-xl font-bold flex items-center gap-2 mb-2">
    <span class="w-2 h-6 bg-[color]-500 rounded-full"></span>
    元件名稱
  </h2>
  <p class="text-sm text-gray-400 mb-6">元件說明</p>

  <!-- 元件實例 -->

  <!-- 狀態顯示區 -->
  <div class="mt-6 p-4 bg-gray-900 rounded-md border border-gray-700">
    <!-- 顯示當前值或 JSON 資料 -->
  </div>
</section>
```

## 特殊注意事項

1. **TypeScript 類型推斷**：當使用 JsonEditor 的 schema prop 時，必須使用 `as const` 確保 type 欄位被推斷為字面類型而非 `string`

2. **搜尋高亮顏色**：所有搜尋匹配項使用 `text-yellow-400` 黃色高亮，這是專案的設計規範

3. **Tailwind 配置**：專案使用 Tailwind 3，CSS @apply 警告可忽略（這是編輯器的 CSS 語言服務問題，不影響建置）

4. **套件管理器**：必須使用 pnpm（版本 10.25.0），這在 package.json 中已指定

5. **Demo vs 套件**：`App.vue` 和 `main.ts` 僅用於開發時測試，不會包含在發布的套件中。套件的入口點是 `src/index.ts`
