# 如何在 VimFront 專案中使用 VimLibrary

## 專案結構

```
VimSharpAI/
├── VimLibrary/          # Web Component Library (這個專案)
│   ├── src/            # 原始碼
│   ├── dist/           # 編譯輸出
│   └── package.json
└── VimFront/           # 前端專案 (將來建立)
    └── package.json
```

## 安裝方式

### 方法 1: 使用本地路徑 (開發階段推薦)

在 VimFront 專案中執行：

```bash
cd VimFront
pnpm add file:../VimLibrary
```

### 方法 2: 發佈到 npm (生產環境)

1. 先發佈 VimLibrary 到 npm：

```bash
cd VimLibrary
npm publish
```

2. 在 VimFront 專案中安裝：

```bash
cd VimFront
pnpm add vim-library
```

### 方法 3: 使用 pnpm workspace (推薦用於 monorepo)

在專案根目錄建立 `pnpm-workspace.yaml`：

```yaml
packages:
  - 'VimLibrary'
  - 'VimFront'
```

然後在 VimFront 的 `package.json` 中：

```json
{
  "dependencies": {
    "vim-library": "workspace:*"
  }
}
```

## 使用範例

### TypeScript/ES6

```typescript
import { VimEditor } from 'vim-library'

// 建立編輯器實例
const editor = new VimEditor()

// 初始化
editor.init({
  initialContent: 'Hello, Vim!',
  config: {
    mode: 'normal',
    lineNumbers: true
  }
})

// 添加到 DOM
document.getElementById('app')?.appendChild(editor)

// 操作編輯器
editor.setContent('New content')
const content = editor.getContent()
```

### HTML (直接使用)

```html
<!DOCTYPE html>
<html>
<head>
  <script type="module">
    import { VimEditor } from 'vim-library'
    
    const editor = document.querySelector('vim-editor')
    editor.init({ initialContent: 'Hello!' })
  </script>
</head>
<body>
  <vim-editor></vim-editor>
</body>
</html>
```

### React

```tsx
import { useEffect, useRef } from 'react'
import { VimEditor } from 'vim-library'

function VimEditorComponent() {
  const editorRef = useRef<VimEditor>(null)

  useEffect(() => {
    if (editorRef.current) {
      editorRef.current.init({
        initialContent: 'Hello from React!'
      })
    }
  }, [])

  return <vim-editor ref={editorRef}></vim-editor>
}
```

### Vue 3

```vue
<template>
  <vim-editor ref="editorRef"></vim-editor>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { VimEditor } from 'vim-library'

const editorRef = ref<VimEditor>()

onMounted(() => {
  editorRef.value?.init({
    initialContent: 'Hello from Vue!'
  })
})
</script>
```

## 型別支援

Library 已包含完整的 TypeScript 型別定義：

```typescript
import type { VimEditor, VimEditorOptions, VimConfig } from 'vim-library'

// 你的 IDE 會提供完整的型別提示和自動完成
```

## 開發流程

1. **在 VimLibrary 中開發**：
   ```bash
   cd VimLibrary
   pnpm dev          # 開發模式
   pnpm build        # 建置
   ```

2. **在 VimFront 中使用**：
   ```bash
   cd VimFront
   pnpm install      # 安裝依賴 (包含 VimLibrary)
   pnpm dev          # 開始開發
   ```

3. **更新 Library**：
   - 修改 VimLibrary 程式碼
   - 執行 `pnpm build`
   - 在 VimFront 中重新載入即可看到變更

## 打包輸出

每次建置會產生以下檔案：

- `dist/vim-library.es.js` - ES Module 格式 (用於現代打包工具)
- `dist/vim-library.umd.js` - UMD 格式 (用於瀏覽器直接引用)
- `dist/index.d.ts` - TypeScript 型別定義

## 版本管理

建議使用語意化版本 (Semantic Versioning)：

- **MAJOR**: 不相容的 API 變更
- **MINOR**: 新增向下相容的功能
- **PATCH**: 向下相容的 Bug 修復

更新版本：
```bash
cd VimLibrary
npm version patch   # 1.0.0 -> 1.0.1
npm version minor   # 1.0.0 -> 1.1.0
npm version major   # 1.0.0 -> 2.0.0
```

