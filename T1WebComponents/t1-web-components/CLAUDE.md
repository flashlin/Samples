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

**3. JsonEditor 設計**

JsonEditor 支援兩種模式，根據輸入的 JSON 字串自動判斷：

- **Array Mode**: JSON 陣列 `[...]` → 顯示為表格，支援搜尋、新增、編輯、刪除、插入操作
- **Object Mode**: JSON 物件 `{...}` → 顯示為表單，需點擊 Save 才更新 modelValue

**Schema 定義約定**：

```typescript
interface JsonSchemaField {
  key: string                          // 欄位鍵名
  label?: string                       // 顯示標籤（可選）
  type: 'string' | 'number' | 'date'  // 欄位類型
}
```

**重要**：必須使用 `as const` 確保類型正確推斷：

```typescript
const schema = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Name', type: 'string' as const }
]
```

**空值處理**：
- Object Mode 接收空字串時，會根據 schema 自動初始化表單欄位
- Array Mode 接收空字串時，顯示空表格

**v-model 綁定**：
- 接受和發出 **JSON 字串**（非物件）
- 使用 `compact` prop 控制輸出格式（`true` = 壓縮單行，`false` = 格式化多行）

**4. 套件建置配置**

這是一個可發布的 npm 套件專案：

- **建置指令**：`pnpm build` 會執行 `vue-tsc && vite build`
- **Vite Library Mode**：建置 UMD 和 ES 模組
- **型別生成**：`vite-plugin-dts` 自動生成 TypeScript 宣告檔
- **Peer Dependency**：Vue 3 被標記為 `external`，不打包進輸出檔案
- **套件入口點**：`src/index.ts` 匯出所有元件和工具函式
- **Demo 檔案**：`App.vue` 和 `main.ts` 僅用於開發測試，不包含在發布套件中

**建置輸出** (dist/)：
```
dist/
├── t1-web-components.es.js    # ES 模組
├── t1-web-components.umd.js   # UMD 模組
└── index.d.ts                 # TypeScript 型別定義
```

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

## 技術約定與注意事項

### 必須遵守

1. **套件管理器**：專案鎖定使用 `pnpm@10.25.0`，不可使用 npm 或 yarn

2. **TypeScript 類型推斷**：JsonEditor 的 schema 必須使用 `as const` 斷言
   ```typescript
   // ✅ 正確
   { key: 'id', type: 'number' as const }

   // ❌ 錯誤 - type 會被推斷為 string
   { key: 'id', type: 'number' }
   ```

3. **JSON 字串綁定**：JsonEditor 的 v-model 接受字串，不是物件
   ```typescript
   // ✅ 正確
   const data = ref<string>('[{"id":1}]')

   // ❌ 錯誤
   const data = ref([{id: 1}])
   ```

### 設計規範

1. **搜尋高亮顏色**：所有匹配項統一使用 `text-yellow-400` 黃色

2. **Tailwind 版本**：使用 Tailwind CSS v3，CSS @apply 語法的編輯器警告可忽略

3. **暗黑模式調色盤**：
   - 主背景：`bg-gray-900`
   - 次級背景：`bg-gray-800/50`
   - 邊框：`border-gray-700`
   - 文字：`text-white`, `text-gray-400`

### 套件使用要求

使用此套件的專案必須：
- 安裝 Vue 3 (peer dependency)
- 安裝並配置 Tailwind CSS v3
- 在 CSS 中引入 Tailwind：
  ```css
  @tailwind base;
  @tailwind components;
  @tailwind utilities;
  ```

## 元件 API 快速參考

### DropDownList

**用途**：強制從選項中選擇，不允許自由輸入

**Props**：
- `modelValue`: `string | number` - v-model 綁定值
- `options`: `Array<{label: string, value: string | number}>` - 選項清單
- `placeholder`: `string` - 輸入框提示文字（選填）
- `inputClass`: `string` - 自訂 CSS 類別（選填）

**Events**：
- `@update:modelValue` - 選擇值變更時觸發

**特色**：Camel Case 搜尋、鍵盤導航（↑↓ + Enter）、黃色高亮

---

### AutoComplete

**用途**：允許自由輸入，提供智慧推薦

**Props**：
- `modelValue`: `string | number` - v-model 綁定值
- `options`: `Array<string | {text: string, value: any}>` - 建議清單
- `placeholder`: `string` - 輸入框提示文字（選填）
- `inputClass`: `string` - 自訂 CSS 類別（選填）

**Events**：
- `@update:modelValue` - 輸入或選擇變更時觸發
- `@change` - 選擇清單項目時觸發，返回項目物件

**特色**：可保留任意輸入、Camel Case 搜尋、黃色高亮

---

### JsonEditor

**用途**：動態 JSON 資料編輯器，支援陣列和物件

**Props**：
- `modelValue`: `string | null` - v-model 綁定的 JSON 字串
- `schema`: `JsonSchemaField[]` - 欄位定義（必須使用 `as const`）
- `compact`: `boolean` - 輸出壓縮格式（預設 `false`）

**Events**：
- `@update:modelValue` - JSON 字串變更時觸發
- `@change` - 資料變更時觸發
- `@error` - JSON 解析錯誤時觸發

**模式自動判斷**：
- 輸入 `[...]` → Array Mode（表格 + 即時儲存）
- 輸入 `{...}` → Object Mode（表單 + 需點 Save）
- 輸入空字串 → 根據 schema 初始化表單

**Array Mode 功能**：搜尋、新增、編輯、刪除、插入、全部刪除

**Object Mode 功能**：表單編輯、Save/Cancel 按鈕
