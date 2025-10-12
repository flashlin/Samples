<template>
  <div class="app-container">
    <h1>Vim Editor Demo - LINQ to T-SQL Converter</h1>
    
    <div class="editor-section">
      <vim-editor
        ref="vimEditorRef"
        :width="editorWidth"
        :height="editorHeight"
        @change="handleEditorChange"
      ></vim-editor>
      
      <div v-if="parseErrors.length > 0" class="error-panel">
        <h3>Parse Errors:</h3>
        <div v-for="(error, index) in parseErrors" :key="index" class="error-message">
          {{ error }}
        </div>
      </div>
      
      <div class="button-container">
        <button @click="executeCode" class="run-button">Run</button>
      </div>
      
      <div class="output-section">
        <h3>T-SQL Output:</h3>
        <textarea 
          v-model="tsqlOutput" 
          class="tsql-textarea"
          readonly
          placeholder="T-SQL output will appear here..."
        ></textarea>
      </div>
    </div>

    <div class="result-section">
      <h2>Execution History</h2>
      <table class="result-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>LINQ Query</th>
            <th>T-SQL Result</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="executionResults.length === 0">
            <td colspan="4" class="empty-message">No results yet. Click "Run" to convert LINQ to T-SQL.</td>
          </tr>
          <tr v-for="(result, index) in executionResults" :key="index">
            <td>{{ result.timestamp }}</td>
            <td class="content-cell">{{ result.content }}</td>
            <td class="result-cell">{{ result.result }}</td>
            <td class="status-cell" :class="result.status">{{ result.status }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import 'vimcomponent'
import { LinqParser, LinqToTSqlConverter, TSqlFormatter } from 'tssql'

interface ExecutionResult {
  timestamp: string
  content: string
  result: string
  status: 'success' | 'error'
}

const editorWidth = '90%'
const editorHeight = '300px'

const editorContent = ref(`FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.age > 18
WHERE u.status = 1
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 0
ORDER BY u.name ASC
SELECT u.name, COUNT(o.id) AS order_count`)

const vimEditorRef = ref<any>(null)
const executionResults = ref<ExecutionResult[]>([])
const tsqlOutput = ref('')
const parseErrors = ref<string[]>([])

const parser = new LinqParser()
const converter = new LinqToTSqlConverter()
const formatter = new TSqlFormatter()

const allTableNameList = ref([
  { name: 'users', description: 'User accounts table' },
  { name: 'orders', description: 'Order records' },
  { name: 'products', description: 'Product catalog' },
  { name: 'customers', description: 'Customer information' },
  { name: 'order_items', description: 'Order line items' },
  { name: 'categories', description: 'Product categories' },
  { name: 'suppliers', description: 'Supplier information' },
  { name: 'employees', description: 'Employee records' },
  { name: 'homes', description: 'Address records' },
  { name: 'friends', description: 'Friendship records' },
])

const handleEditorChange = (event: CustomEvent) => {
  editorContent.value = event.detail.content.join('\n')
}

const extractCurrentWord = (lineBeforeCursor: string): string => {
  const match = lineBeforeCursor.match(/[a-zA-Z0-9]*$/)
  return match ? match[0] : ''
}

const detectSqlContext = (beforeCursor: string): string => {
  const keywords = ['FROM', 'SELECT', 'WHERE', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'ORDER BY', 'GROUP BY', 'HAVING']
  const upperText = beforeCursor.toUpperCase()
  
  let lastKeyword = ''
  let lastIndex = -1
  
  for (const keyword of keywords) {
    const index = upperText.lastIndexOf(keyword)
    if (index > lastIndex) {
      lastIndex = index
      lastKeyword = keyword
    }
  }
  
  if (lastKeyword.includes('FROM') || lastKeyword.includes('JOIN')) {
    return 'FROM'
  } else if (lastKeyword === 'SELECT') {
    return 'SELECT'
  } else if (lastKeyword === 'WHERE' || lastKeyword === 'HAVING') {
    return 'WHERE'
  } else if (lastKeyword.includes('ORDER') || lastKeyword.includes('GROUP')) {
    return 'ORDER'
  }
  
  return 'FROM'
}

const generateSuggestions = (context: string, currentWord: string): any[] => {
  if (context === 'FROM') {
    const filtered = allTableNameList.value.filter(table =>
      table.name.toLowerCase().startsWith(currentWord.toLowerCase())
    )
    
    return filtered.map(table => ({
      text: table.name,
      description: table.description,
      action: () => replaceCurrentWord(table.name, currentWord)
    }))
  }
  
  return []
}

const replaceCurrentWord = (newText: string, oldWord: string) => {
  const editor = vimEditorRef.value
  if (!editor) return
  
  editor.replaceWordAtCursor(oldWord, newText)
}

const handleIntellisense = (event: CustomEvent<any>) => {
  const ctx = event.detail
  
  const beforeCursor = ctx.contentBeforeCursor
  const currentWord = extractCurrentWord(ctx.lineBeforeCursor)
  const context = detectSqlContext(beforeCursor)
  const suggestions = generateSuggestions(context, currentWord)
  
  if (suggestions.length > 0) {
    vimEditorRef.value?.showIntellisense(suggestions)
  }
}

const executeCode = () => {
  const timestamp = new Date().toLocaleString()
  const linq = editorContent.value
  
  try {
    const parseResult = parser.parse(linq)
    const tsqlQuery = converter.convert(parseResult.result)
    const sql = formatter.format(tsqlQuery)
    const errors = parseResult.errors.map((e: any) => e.message)
    
    tsqlOutput.value = sql
    parseErrors.value = errors
    
    executionResults.value.unshift({
      timestamp,
      content: linq.substring(0, 80) + (linq.length > 80 ? '...' : ''),
      result: sql,
      status: errors.length > 0 ? 'error' : 'success'
    })
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error)
    tsqlOutput.value = ''
    parseErrors.value = [errorMsg]
    
    executionResults.value.unshift({
      timestamp,
      content: linq.substring(0, 80) + (linq.length > 80 ? '...' : ''),
      result: errorMsg,
      status: 'error'
    })
  }
}

onMounted(() => {
  customElements.whenDefined('vim-editor').then(() => {
    const editor = vimEditorRef.value
    if (editor) {
      editor.load(editorContent.value)
      editor.addEventListener('intellisense', handleIntellisense)
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

h3 {
  color: #555;
  margin-bottom: 10px;
  font-size: 16px;
  font-weight: 600;
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

.error-panel {
  background-color: #fee;
  border: 2px solid #fcc;
  border-radius: 4px;
  padding: 15px;
  margin-bottom: 15px;
}

.error-panel h3 {
  color: #c00;
  margin-top: 0;
  margin-bottom: 10px;
}

.error-message {
  color: #c00;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  padding: 5px 0;
}

.button-container {
  display: flex;
  justify-content: flex-start;
  gap: 10px;
  margin-bottom: 20px;
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

.output-section {
  margin-top: 20px;
}

.tsql-textarea {
  width: 100%;
  min-height: 200px;
  padding: 15px;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  border: 2px solid #ddd;
  border-radius: 4px;
  resize: vertical;
  background-color: #f8f9fa;
  color: #333;
  white-space: pre;
  overflow: auto;
}

.tsql-textarea:focus {
  outline: none;
  border-color: #4CAF50;
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
  max-width: 250px;
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
  max-width: 350px;
}

.status-cell {
  font-weight: bold;
  text-transform: uppercase;
  font-size: 12px;
}

.status-cell.success {
  color: #28a745;
}

.status-cell.error {
  color: #dc3545;
}

.empty-message {
  text-align: center;
  color: #999;
  font-style: italic;
  padding: 30px !important;
}
</style>

