<script setup lang="ts">
import { ref } from 'vue'
import { convertTableFormatToCsv } from '../tools/textTool'
defineProps<{ msg: string }>()

const code = ref('// 在這裡輸入您的程式碼\nfunction hello() {\n  console.log("Hello, World!");\n}')
const clipboardError = ref('')

function clickConvertTableFormatToCsv() {
  code.value = convertTableFormatToCsv(code.value)
}

async function copyFromClipboard() {
  try {
    const permissionStatus = await navigator.permissions.query({ name: 'clipboard-read' as PermissionName });
    if (permissionStatus.state === 'granted' || permissionStatus.state === 'prompt') {
      const text = await navigator.clipboard.readText();
      code.value = text;
      clipboardError.value = '';
    } else {
      clipboardError.value = '沒有剪貼簿讀取權限';
      useAlternativeClipboardMethod();
    }
  } catch (error) {
    clipboardError.value = '讀取剪貼簿失敗，請嘗試手動貼上';
    useAlternativeClipboardMethod();
  }
}

function useAlternativeClipboardMethod() {
  const textArea = document.createElement('textarea');
  document.body.appendChild(textArea);
  textArea.focus();
  
  try {
    const successful = document.execCommand('paste');
    if (successful) {
      code.value = textArea.value;
      clipboardError.value = '';
    } else {
      clipboardError.value = '請使用 Ctrl+V 手動貼上';
    }
  } catch (err) {
    clipboardError.value = '請使用 Ctrl+V 手動貼上';
  }
  
  document.body.removeChild(textArea);
}

function handleCodeChange(event: Event) {
  const target = event.target as HTMLTextAreaElement
  code.value = target.value
}

function runCode() {
  try {
    // 使用 Function 構造函數來執行程式碼
    // 注意：這在生產環境中可能存在安全風險
    const result = new Function(code.value)()
    alert('程式碼執行成功！')
    console.log('執行結果:', result)
  } catch (error) {
    alert(`程式碼執行錯誤: ${error}`)
    console.error('執行錯誤:', error)
  }
}
</script>

<template>
  <h1>{{ msg }}</h1>

  <div class="card">
    <button type="button" @click="copyFromClipboard">從剪貼簿複製</button>
    <button type="button" @click="clickConvertTableFormatToCsv">ToCsv</button>
    <p v-if="clipboardError" class="error-message">{{ clipboardError }}</p>
    <p>
      Edit
      <code>components/HelloWorld.vue</code> to test HMR
    </p>
  </div>

  <div class="editor-container">
    <h2>程式碼編輯器</h2>
    <textarea 
      class="code-editor" 
      v-model="code" 
      @input="handleCodeChange"
      spellcheck="false"
    ></textarea>
    <div class="editor-actions">
      <button @click="runCode" class="run-button">執行程式碼</button>
    </div>
  </div>

  <p>
    Check out
    <a href="https://vuejs.org/guide/quick-start.html#local" target="_blank"
      >create-vue</a
    >, the official Vue + Vite starter
  </p>
  <p>
    Learn more about IDE Support for Vue in the
    <a
      href="https://vuejs.org/guide/scaling-up/tooling.html#ide-support"
      target="_blank"
      >Vue Docs Scaling up Guide</a
    >.
  </p>
  <p class="read-the-docs">Click on the Vite and Vue logos to learn more</p>
  
</template>

<style scoped>
.read-the-docs {
  color: #888;
}

.editor-container {
  margin: 20px 0;
  width: 100%;
  max-width: 100%;
}

.editor-container h2 {
  margin-bottom: 10px;
}

.code-editor {
  width: 100%;
  height: 400px;
  font-family: 'Courier New', Courier, monospace;
  font-size: 14px;
  line-height: 1.5;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #1e1e1e;
  color: #d4d4d4;
  resize: vertical;
  tab-size: 2;
  overflow: auto;
}

.editor-actions {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
}

.run-button {
  background-color: #4CAF50;
  color: white;
  border: none;
  padding: 8px 16px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 14px;
  border-radius: 4px;
  cursor: pointer;
}

.run-button:hover {
  background-color: #45a049;
}

.error-message {
  color: #f44336;
  margin-top: 5px;
  font-size: 14px;
}

@media (min-width: 768px) {
  .card {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }
  
  .code-editor {
    height: 500px;
  }
}
</style>
