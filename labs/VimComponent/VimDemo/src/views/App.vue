<template>
  <div class="app-container">
    <h1>Vim Editor Demo</h1>
    
    <div class="editor-section">
      <vim-editor
        ref="vimEditorRef"
        :width="editorWidth"
        :height="editorHeight"
        @change="handleEditorChange"
      ></vim-editor>
      
      <div class="button-container">
        <button @click="executeCode" class="run-button">Run</button>
      </div>
    </div>

    <div class="result-section">
      <h2>Execution Results</h2>
      <table class="result-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Content</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="executionResults.length === 0">
            <td colspan="3" class="empty-message">No results yet. Click "Run" to execute code.</td>
          </tr>
          <tr v-for="(result, index) in executionResults" :key="index">
            <td>{{ result.timestamp }}</td>
            <td class="content-cell">{{ result.content }}</td>
            <td class="result-cell" :class="result.status">{{ result.result }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

// Import the VimComponent package
import 'vimcomponent'

interface ExecutionResult {
  timestamp: string
  content: string
  result: string
  status: 'success' | 'error'
}

// Editor configuration
const editorWidth = '90%'
const editorHeight = '300px'

// Editor content as a single string
const editorContent = ref(`// Enter your code here
console.log('Hello from Vim Editor!');

function greet(name: string) {
  return \`Hello, \${name}!\`;
}

greet('World');`)

// Editor ref
const vimEditorRef = ref<any>(null)

// Execution results
const executionResults = ref<ExecutionResult[]>([])

// Handle editor content change
const handleEditorChange = (event: CustomEvent) => {
  // event.detail.content is string[], join back to string
  editorContent.value = event.detail.content.join('\n')
}

// Execute code function
const executeCode = () => {
  const timestamp = new Date().toLocaleString()
  const codeString = editorContent.value
  
  try {
    // Capture console.log output
    const logs: string[] = []
    const originalLog = console.log
    console.log = (...args: any[]) => {
      logs.push(args.join(' '))
      originalLog.apply(console, args)
    }

    // Execute the code
    const result = eval(codeString)
    
    // Restore console.log
    console.log = originalLog
    
    // Add result to table
    executionResults.value.unshift({
      timestamp,
      content: codeString.substring(0, 100) + (codeString.length > 100 ? '...' : ''),
      result: logs.length > 0 ? logs.join('\n') : (result !== undefined ? String(result) : 'Executed successfully'),
      status: 'success'
    })
  } catch (error) {
    // Add error to table
    executionResults.value.unshift({
      timestamp,
      content: codeString.substring(0, 100) + (codeString.length > 100 ? '...' : ''),
      result: error instanceof Error ? error.message : String(error),
      status: 'error'
    })
  }
}

onMounted(() => {
  // Wait for the custom element to be defined
  customElements.whenDefined('vim-editor').then(() => {
    console.log('vim-editor component is ready')
    
    // Load initial content using the load() method
    const editor = vimEditorRef.value
    if (editor) {
      editor.load(editorContent.value)
    }
  })
})
</script>

<style scoped>
.app-container {
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

h1 {
  color: #333;
  margin-bottom: 20px;
  font-size: 28px;
}

h2 {
  color: #555;
  margin-bottom: 15px;
  font-size: 20px;
}

.editor-section {
  margin-bottom: 30px;
}

vim-editor {
  display: block;
  margin-bottom: 15px;
  border: 2px solid #ddd;
  border-radius: 4px;
  overflow: hidden;
}

.button-container {
  display: flex;
  justify-content: flex-start;
  gap: 10px;
}

.run-button {
  padding: 10px 30px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.3s;
}

.run-button:hover {
  background-color: #45a049;
}

.run-button:active {
  transform: translateY(1px);
}

.result-section {
  margin-top: 30px;
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  background-color: white;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border-radius: 4px;
  overflow: hidden;
}

.result-table thead {
  background-color: #f8f9fa;
}

.result-table th {
  padding: 12px;
  text-align: left;
  font-weight: 600;
  color: #555;
  border-bottom: 2px solid #dee2e6;
}

.result-table td {
  padding: 12px;
  border-bottom: 1px solid #dee2e6;
  color: #333;
}

.result-table tbody tr:hover {
  background-color: #f8f9fa;
}

.result-table tbody tr:last-child td {
  border-bottom: none;
}

.content-cell {
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: 'Courier New', monospace;
  font-size: 13px;
}

.result-cell {
  font-family: 'Courier New', monospace;
  font-size: 13px;
  white-space: pre-wrap;
  max-width: 400px;
}

.result-cell.success {
  color: #28a745;
}

.result-cell.error {
  color: #dc3545;
}

.empty-message {
  text-align: center;
  color: #999;
  font-style: italic;
  padding: 30px !important;
}
</style>

