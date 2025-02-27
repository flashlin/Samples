<script setup lang="ts">
import { ref } from 'vue'

defineProps<{ msg: string }>()

const count = ref(0)
const code = ref('// 在這裡輸入您的程式碼\nfunction hello() {\n  console.log("Hello, World!");\n}')

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
    <button type="button" @click="count++">count is {{ count }}</button>
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
}

.editor-container h2 {
  margin-bottom: 10px;
}

.code-editor {
  width: 100%;
  height: 300px;
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
</style>
