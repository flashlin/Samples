# VimComponent 使用指南

## VimComponent 屬性說明

### content 屬性

**重要**：VimComponent 的 `content` 屬性期望的是 **字串陣列（`string[]`）**，每個元素代表一行。

#### ❌ 錯誤用法

```vue
<template>
  <vim-editor :content="editorContent" />
</template>

<script setup>
const editorContent = ref(`line 1
line 2
line 3`)
// 這會導致每個字元被當作一行！
</script>
```

**問題**：傳入單一字串會導致每個字元被當作一行顯示：
```
l
i
n
e
 
1
...
```

#### ✅ 正確用法

```vue
<template>
  <vim-editor :content="editorContent" />
</template>

<script setup>
const editorContent = ref<string[]>([
  'line 1',
  'line 2',
  'line 3'
])
// 正確：陣列的每個元素是一行
</script>
```

---

## 完整範例

### 基本使用

```vue
<template>
  <vim-editor
    :width="'90%'"
    :height="'300px'"
    :content="content"
    @change="handleChange"
  />
</template>

<script setup lang="ts">
import { ref } from 'vue'
import 't1-vim-editor'

// Content as array of lines
const content = ref<string[]>([
  '// JavaScript code',
  'console.log("Hello World!");',
  '',
  'function add(a, b) {',
  '  return a + b;',
  '}'
])

// Handle content change
const handleChange = (event: CustomEvent) => {
  // event.detail.content is string[]
  content.value = event.detail.content
  console.log('Content changed:', content.value)
}
</script>
```

### 從字串轉換

如果你有一個多行字串，需要先分割成陣列：

```typescript
// 從字串轉換
const codeString = `// Line 1
// Line 2
console.log('test');`

const content = ref<string[]>(codeString.split('\n'))
```

### 轉換回字串

執行或保存時，需要將陣列合併回字串：

```typescript
// 陣列轉回字串
const codeString = content.value.join('\n')

// 用於執行
eval(codeString)

// 或保存
localStorage.setItem('code', codeString)
```

---

## 屬性列表

| 屬性 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `width` | `string` | `'800px'` | 編輯器寬度 |
| `height` | `string` | `'600px'` | 編輯器高度 |
| `content` | `string[]` | `[]` | **內容（行陣列）** |

---

## 事件列表

| 事件 | 參數 | 說明 |
|------|------|------|
| `change` | `CustomEvent<{ content: string[] }>` | 內容變更時觸發 |

**事件處理範例**：

```typescript
const handleChange = (event: CustomEvent) => {
  const newContent = event.detail.content  // string[]
  console.log('Lines:', newContent.length)
  console.log('Content:', newContent.join('\n'))
}
```

---

## 方法

### getStatus()

獲取編輯器狀態：

```typescript
const editor = document.querySelector('vim-editor')
const status = editor.getStatus()

console.log(status)
// {
//   mode: 'normal' | 'insert' | 'visual' | ...,
//   cursorX: number,
//   cursorY: number,
//   cursorVisible: boolean
// }
```

### getBuffer()

獲取緩衝區內容：

```typescript
const buffer = editor.getBuffer()
// BufferCell[][]
```

### setContent(content: string[])

程式化設定內容：

```typescript
const editor = document.querySelector('vim-editor')
editor.setContent([
  'New line 1',
  'New line 2'
])
```

### load(text: string)

載入文字（自動分割成行）：

```typescript
const editor = document.querySelector('vim-editor')
editor.load(`Line 1
Line 2
Line 3`)
```

---

## 常見模式

### 1. 載入檔案

```typescript
const loadFile = async (file: File) => {
  const text = await file.text()
  const lines = text.split('\n')
  content.value = lines
}
```

### 2. 保存檔案

```typescript
const saveFile = () => {
  const text = content.value.join('\n')
  const blob = new Blob([text], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  
  const a = document.createElement('a')
  a.href = url
  a.download = 'code.txt'
  a.click()
}
```

### 3. 與 LocalStorage 整合

```typescript
// 保存
const saveToStorage = () => {
  const text = content.value.join('\n')
  localStorage.setItem('editor-content', text)
}

// 載入
const loadFromStorage = () => {
  const text = localStorage.getItem('editor-content')
  if (text) {
    content.value = text.split('\n')
  }
}
```

### 4. 執行程式碼

```typescript
const executeCode = () => {
  // 合併成單一字串
  const codeString = content.value.join('\n')
  
  try {
    const result = eval(codeString)
    console.log('Result:', result)
  } catch (error) {
    console.error('Error:', error)
  }
}
```

---

## TypeScript 類型定義

```typescript
// Content type
type EditorContent = string[]

// Status type
interface EditorStatus {
  mode: 'normal' | 'insert' | 'visual' | 'visual-line' | 'fast-jump' | 'match' | 'search' | 'multi-insert'
  cursorX: number
  cursorY: number
  cursorVisible: boolean
  searchKeyword?: string
  searchMatchCount?: number
}

// Change event
interface ChangeEvent extends CustomEvent {
  detail: {
    content: string[]
  }
}

// Component ref
const editorRef = ref<HTMLElement & {
  getStatus(): EditorStatus
  getBuffer(): BufferCell[][]
  setContent(content: string[]): void
  load(text: string): void
}>()
```

---

## 完整 App.vue 範例

```vue
<template>
  <div class="editor-container">
    <vim-editor
      ref="editorRef"
      :width="'90%'"
      :height="'300px'"
      :content="editorContent"
      @change="handleChange"
    />
    
    <div class="controls">
      <button @click="executeCode">Run</button>
      <button @click="saveCode">Save</button>
      <button @click="loadCode">Load</button>
    </div>
    
    <div class="output">
      <h3>Output:</h3>
      <pre>{{ output }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import 't1-vim-editor'

// Editor content (array of lines)
const editorContent = ref<string[]>([
  '// JavaScript code here',
  'console.log("Hello World!");',
  '',
  'function greet(name) {',
  '  return `Hello, ${name}!`;',
  '}',
  '',
  'const result = greet("Vue");',
  'console.log(result);'
])

const editorRef = ref<any>(null)
const output = ref('')

// Handle content change
const handleChange = (event: CustomEvent) => {
  editorContent.value = event.detail.content
}

// Execute code
const executeCode = () => {
  // Convert array to string
  const codeString = editorContent.value.join('\n')
  
  // Capture console output
  const logs: string[] = []
  const originalLog = console.log
  
  console.log = (...args: any[]) => {
    logs.push(args.join(' '))
    originalLog.apply(console, args)
  }
  
  try {
    eval(codeString)
    output.value = logs.join('\n')
  } catch (error) {
    output.value = `Error: ${error}`
  } finally {
    console.log = originalLog
  }
}

// Save code
const saveCode = () => {
  const text = editorContent.value.join('\n')
  localStorage.setItem('saved-code', text)
  alert('Code saved!')
}

// Load code
const loadCode = () => {
  const text = localStorage.getItem('saved-code')
  if (text) {
    editorContent.value = text.split('\n')
  }
}
</script>

<style scoped>
.editor-container {
  padding: 20px;
}

.controls {
  margin: 20px 0;
  display: flex;
  gap: 10px;
}

button {
  padding: 10px 20px;
  background: #42b883;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background: #35495e;
}

.output {
  margin-top: 20px;
  padding: 10px;
  background: #f5f5f5;
  border-radius: 4px;
}

pre {
  margin: 0;
  font-family: monospace;
}
</style>
```

---

## 疑難排解

### 問題：內容顯示為單個字元

**症狀**：每個字元單獨顯示在一行

**原因**：傳入的是字串而非字串陣列

**解決方案**：
```typescript
// ❌ 錯誤
const content = ref('line 1\nline 2')

// ✅ 正確
const content = ref<string[]>(['line 1', 'line 2'])
```

### 問題：change 事件未觸發

**檢查**：
1. 確認已正確綁定事件：`@change="handleChange"`
2. 確認處理函數簽名正確
3. 在 Vim 的 insert mode 中修改內容

### 問題：無法程式化修改內容

**解決方案**：使用 `setContent` 方法或直接修改 ref

```typescript
// 方法 1: 修改 ref
editorContent.value = ['new', 'content']

// 方法 2: 使用 setContent
const editor = editorRef.value
if (editor) {
  editor.setContent(['new', 'content'])
}
```

---

## 最佳實踐

1. ✅ **總是使用 `string[]` 類型**
2. ✅ **使用 TypeScript 類型註解**
3. ✅ **處理空行（空字串）**
4. ✅ **執行前合併陣列**
5. ✅ **保存前合併陣列**
6. ✅ **載入後分割字串**

---

## 相關資源

- [VimComponent README](../../VimComponent/README.md)
- [ES Module Guide](../../ES_MODULE_GUIDE.md)
- [Troubleshooting Guide](../../TROUBLESHOOTING.md)

