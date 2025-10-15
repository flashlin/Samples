# VimComponent 載入內容的方法

VimComponent 提供兩種方式來設定編輯器內容。

## 方法比較

| 方法 | 資料格式 | 使用時機 | 優點 | 缺點 |
|------|---------|---------|------|------|
| **`:content` 屬性** | `string[]` | 響應式資料綁定 | Vue 響應式更新 | 需要陣列格式 |
| **`load()` 方法** | `string` | 程式化載入 | 接受字串 | 需要手動呼叫 |

---

## 方法 1：使用 `:content` 屬性（響應式）

### 優點
- ✅ Vue 響應式資料綁定
- ✅ 資料變更自動更新編輯器
- ✅ 雙向同步（配合 `@change` 事件）

### 缺點
- ❌ 必須使用 `string[]` 格式
- ❌ 需要手動分割/合併字串

### 範例

```vue
<template>
  <vim-editor
    :content="editorContent"
    @change="handleChange"
  />
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import '@mrbrain/t1-vim-editor'

// Content as array of lines
const editorContent = ref<string[]>([
  '// Line 1',
  '// Line 2',
  'console.log("Hello");'
])

// Handle changes
const handleChange = (event: CustomEvent) => {
  editorContent.value = event.detail.content
}

// Watch for external changes
watch(someOtherData, (newValue) => {
  // Convert string to array
  editorContent.value = newValue.split('\n')
})
</script>
```

---

## 方法 2：使用 `load()` 方法（推薦）✨

### 優點
- ✅ 接受單一字串（自動分割）
- ✅ 簡單直觀
- ✅ 適合初始化和程式化載入
- ✅ 不需要手動分割字串

### 缺點
- ❌ 非響應式（需要手動呼叫）
- ❌ 需要取得 ref

### 範例（目前 VimDemo 使用的方法）

```vue
<template>
  <vim-editor
    ref="vimEditorRef"
    @change="handleEditorChange"
  />
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import '@mrbrain/t1-vim-editor'

// Content as single string (easier to work with)
const editorContent = ref(`// Enter your code here
console.log('Hello from Vim Editor!');

function greet(name: string) {
  return \`Hello, \${name}!\`;
}

greet('World');`)

const vimEditorRef = ref<any>(null)

// Handle editor content change
const handleEditorChange = (event: CustomEvent) => {
  // Convert array back to string
  editorContent.value = event.detail.content.join('\n')
}

// Load content on mount
onMounted(() => {
  customElements.whenDefined('vim-editor').then(() => {
    const editor = vimEditorRef.value
    if (editor) {
      // Load method accepts string and splits automatically
      editor.load(editorContent.value)
    }
  })
})
</script>
```

---

## 使用場景

### 使用 `:content` 屬性

**適合**：
- 需要響應式更新
- 內容頻繁變更
- 需要與其他 Vue 資料同步

```vue
<script setup>
// Example: Auto-save functionality
const content = ref<string[]>(['// Code here'])

watch(content, (newContent) => {
  localStorage.setItem('code', newContent.join('\n'))
}, { deep: true })
</script>
```

### 使用 `load()` 方法

**適合**：
- 初始化時載入
- 從檔案載入
- 從 API 載入
- 重置編輯器內容
- 程式化操作

```vue
<script setup>
// Example: Load from file
const loadFile = async (file: File) => {
  const text = await file.text()
  const editor = vimEditorRef.value
  if (editor) {
    editor.load(text)  // Easy! Just pass the string
  }
}

// Example: Reset
const resetEditor = () => {
  const editor = vimEditorRef.value
  if (editor) {
    editor.load('// Start fresh\n')
  }
}

// Example: Load from API
const loadFromAPI = async (id: string) => {
  const response = await fetch(`/api/code/${id}`)
  const code = await response.text()
  
  const editor = vimEditorRef.value
  if (editor) {
    editor.load(code)
  }
}
</script>
```

---

## 完整對比範例

### 方法 1：`:content` 屬性

```vue
<template>
  <vim-editor :content="lines" @change="handleChange" />
  <button @click="updateContent">Update</button>
</template>

<script setup>
const lines = ref<string[]>(['line 1', 'line 2'])

const handleChange = (e: CustomEvent) => {
  lines.value = e.detail.content
}

const updateContent = () => {
  // Must convert string to array
  const newCode = '// New code\nconsole.log("test");'
  lines.value = newCode.split('\n')
}
</script>
```

### 方法 2：`load()` 方法

```vue
<template>
  <vim-editor ref="editor" @change="handleChange" />
  <button @click="updateContent">Update</button>
</template>

<script setup>
const editor = ref<any>(null)
const content = ref('line 1\nline 2')

const handleChange = (e: CustomEvent) => {
  content.value = e.detail.content.join('\n')
}

const updateContent = () => {
  // Can use string directly
  const newCode = '// New code\nconsole.log("test");'
  content.value = newCode
  editor.value?.load(newCode)
}

onMounted(() => {
  customElements.whenDefined('vim-editor').then(() => {
    editor.value?.load(content.value)
  })
})
</script>
```

---

## 混合使用

你也可以結合兩種方法：

```vue
<template>
  <vim-editor
    ref="editor"
    :content="lines"
    @change="handleChange"
  />
  <button @click="loadFromFile">Load File</button>
  <button @click="reset">Reset</button>
</template>

<script setup>
const editor = ref<any>(null)
const lines = ref<string[]>(['// Initial'])

// Use :content for reactive updates
const handleChange = (e: CustomEvent) => {
  lines.value = e.detail.content
}

// Use load() for programmatic operations
const loadFromFile = async (file: File) => {
  const text = await file.text()
  editor.value?.load(text)
}

const reset = () => {
  editor.value?.load('// Fresh start\n')
}
</script>
```

---

## 最佳實踐建議

### ✅ 推薦：使用 `load()` 方法（VimDemo 採用）

**理由**：
1. **簡單直觀**：直接使用字串，不需要陣列轉換
2. **符合直覺**：大多數情況下程式碼是字串格式
3. **易於整合**：與檔案系統、API、LocalStorage 等配合良好
4. **減少轉換**：不需要頻繁 `split()` 和 `join()`

**模式**：
```typescript
// Store as string
const code = ref('...')

// Load with load()
onMounted(() => {
  customElements.whenDefined('vim-editor').then(() => {
    vimEditorRef.value?.load(code.value)
  })
})

// Sync changes
const handleChange = (e: CustomEvent) => {
  code.value = e.detail.content.join('\n')
}
```

### 當使用 `:content` 屬性時

只在以下情況使用：
- 需要 Vue 的響應式系統自動更新
- 已經有 `string[]` 格式的資料
- 需要與其他陣列資料同步

---

## VimComponent API 參考

### load(text: string)

載入文字內容到編輯器。

**參數**：
- `text` (string): 要載入的文字內容

**行為**：
- 自動使用 `\n` 分割成行
- 重置游標到 (0, 0)
- 清除滾動偏移
- 設定為 normal mode

**範例**：
```typescript
const editor = document.querySelector('vim-editor')
editor.load('line 1\nline 2\nline 3')
```

### setContent(content: string[])

設定編輯器內容（陣列格式）。

**參數**：
- `content` (string[]): 行陣列

**行為**：
- 直接設定內容
- 不改變游標位置
- 不改變模式

**範例**：
```typescript
const editor = document.querySelector('vim-editor')
editor.setContent(['line 1', 'line 2', 'line 3'])
```

### content 屬性

**類型**：`string[]`

可以透過 Vue 的 `:content` 綁定。

**範例**：
```vue
<vim-editor :content="['line 1', 'line 2']" />
```

---

## 總結

| 需求 | 推薦方法 | 原因 |
|------|---------|------|
| 初始化載入 | `load()` | 簡單、接受字串 |
| 從檔案載入 | `load()` | 不需要分割 |
| 從 API 載入 | `load()` | 直接使用回應 |
| 響應式更新 | `:content` | 自動同步 |
| 程式化操作 | `load()` | 更靈活 |
| 重置內容 | `load()` | 自動重置狀態 |

**VimDemo 採用 `load()` 方法，這是大多數情況下的最佳選擇！** ✨

