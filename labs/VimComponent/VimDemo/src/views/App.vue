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
  { 
    name: 'users', 
    description: 'User accounts table',
    fields: ['id', 'name', 'email', 'age', 'status', 'created_at', 'updated_at']
  },
  { 
    name: 'orders', 
    description: 'Order records',
    fields: ['id', 'user_id', 'order_date', 'total_amount', 'status', 'shipping_address']
  },
  { 
    name: 'products', 
    description: 'Product catalog',
    fields: ['id', 'name', 'description', 'price', 'stock', 'category_id']
  },
  { 
    name: 'customers', 
    description: 'Customer information',
    fields: ['id', 'first_name', 'last_name', 'email', 'phone', 'address']
  },
  { 
    name: 'order_items', 
    description: 'Order line items',
    fields: ['id', 'order_id', 'product_id', 'quantity', 'unit_price', 'subtotal']
  },
  { 
    name: 'categories', 
    description: 'Product categories',
    fields: ['id', 'name', 'description', 'parent_id']
  },
  { 
    name: 'suppliers', 
    description: 'Supplier information',
    fields: ['id', 'company_name', 'contact_name', 'email', 'phone', 'address']
  },
  { 
    name: 'employees', 
    description: 'Employee records',
    fields: ['id', 'first_name', 'last_name', 'email', 'department', 'salary', 'hire_date']
  },
  { 
    name: 'homes', 
    description: 'Address records',
    fields: ['id', 'street', 'city', 'state', 'zip_code', 'country']
  },
  { 
    name: 'friends', 
    description: 'Friendship records',
    fields: ['id', 'user_id', 'friend_id', 'status', 'created_at']
  },
])

const handleEditorChange = (event: CustomEvent) => {
  editorContent.value = event.detail.content.join('\n')
}

const extractCurrentWord = (lineBeforeCursor: string): string => {
  const match = lineBeforeCursor.match(/[a-zA-Z0-9_]*$/)
  return match ? match[0] : ''
}

const extractTableAliasMap = (content: string): Map<string, string> => {
  const aliasMap = new Map<string, string>()
  
  try {
    const parseResult = parser.parse(content)
    const query = parseResult.result
    
    if (query && query.from) {
      const tableName = query.from.tableName
      const alias = query.from.alias || tableName
      aliasMap.set(alias, tableName)
      if (alias !== tableName) {
        aliasMap.set(tableName, tableName)
      }
    }
    
    if (query && query.joins) {
      query.joins.forEach((join: any) => {
        const tableName = join.tableName
        const alias = join.alias || tableName
        aliasMap.set(alias, tableName)
        if (alias !== tableName) {
          aliasMap.set(tableName, tableName)
        }
      })
    }
  } catch (e) {
    console.log('[Intellisense] Parse error:', e)
  }
  
  return aliasMap
}

const detectIntellisenseContext = (beforeCursor: string, content: string): {
  type: 'table' | 'field' | 'field-with-table',
  tableName?: string,
  currentWord: string
} => {
  const upperText = beforeCursor.toUpperCase()
  const currentWord = extractCurrentWord(beforeCursor)
  
  const fromMatch = upperText.match(/FROM\s+([a-zA-Z0-9_]*)\s*$/i)
  if (fromMatch) {
    return { type: 'table', currentWord }
  }
  
  const joinMatch = upperText.match(/(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|JOIN)\s+([a-zA-Z0-9_]*)\s*$/i)
  if (joinMatch) {
    return { type: 'table', currentWord }
  }
  
  const aliasMap = extractTableAliasMap(content)
  
  const fieldWithAliasMatch = beforeCursor.match(/([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]*)$/)
  if (fieldWithAliasMatch) {
    const alias = fieldWithAliasMatch[1]
    const fieldWord = fieldWithAliasMatch[2]
    const tableName = aliasMap.get(alias) || alias
    
    return { 
      type: 'field-with-table', 
      tableName,
      currentWord: fieldWord
    }
  }
  
  const selectMatch = upperText.match(/SELECT\s+[^FROM]*$/i)
  if (selectMatch) {
    return { type: 'field', currentWord }
  }
  
  const whereMatch = upperText.match(/(WHERE|HAVING|AND|OR)\s+[^FROM]*$/i)
  if (whereMatch) {
    return { type: 'field', currentWord }
  }
  
  const orderMatch = upperText.match(/(ORDER\s+BY|GROUP\s+BY)\s+[^FROM]*$/i)
  if (orderMatch) {
    return { type: 'field', currentWord }
  }
  
  return { type: 'table', currentWord }
}

const generateSuggestions = (context: any): any[] => {
  const suggestions: any[] = []
  
  if (context.type === 'table') {
    const filtered = allTableNameList.value.filter(table =>
      table.name.toLowerCase().startsWith(context.currentWord.toLowerCase())
    )
    
    return filtered.map(table => ({
      text: table.name,
      description: table.description,
      action: () => replaceCurrentWord(table.name, context.currentWord)
    }))
  }
  
  if (context.type === 'field-with-table' && context.tableName) {
    const table = allTableNameList.value.find(t => 
      t.name.toLowerCase() === context.tableName?.toLowerCase()
    )
    
    if (table) {
      const filtered = table.fields.filter(field =>
        field.toLowerCase().startsWith(context.currentWord.toLowerCase())
      )
      
      return filtered.map(field => ({
        text: field,
        description: `${context.tableName}.${field}`,
        action: () => replaceCurrentWord(field, context.currentWord)
      }))
    }
  }
  
  if (context.type === 'field') {
    allTableNameList.value.forEach(table => {
      const filtered = table.fields.filter(field =>
        field.toLowerCase().startsWith(context.currentWord.toLowerCase())
      )
      
      filtered.forEach(field => {
        suggestions.push({
          text: field,
          description: `${table.name}.${field}`,
          action: () => replaceCurrentWord(field, context.currentWord)
        })
      })
    })
    
    return suggestions
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
  
  console.log('[Intellisense] Context:', ctx)
  
  const fullContent = ctx.contentBeforeCursor + ctx.contentAfterCursor
  const context = detectIntellisenseContext(ctx.contentBeforeCursor, fullContent)
  
  console.log('[Intellisense] Detected context:', context)
  
  const suggestions = generateSuggestions(context)
  
  console.log('[Intellisense] Generated suggestions:', suggestions.length)
  
  if (suggestions.length > 0) {
    vimEditorRef.value?.showIntellisense(suggestions)
  } else {
    console.log('[Intellisense] No suggestions found')
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

