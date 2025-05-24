<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { convertTableFormatToCsv, convertJsonFormatToCsv, convertCsvFormatToJson, 
  convertCsvFormatToTable, convertCsvFormatToSql, getCsvHeadersName, cutCsvText } from '../tools/textTool'
import { copyFromClipboard, pasteToClipboard } from '../tools/clipboardTool'
import { translateToEnAsync, translateToZhAsync } from '../tools/translateApi'
import Loading from './Loading.vue'

defineProps<{ msg: string }>()

const code = ref(`id name age
1 flash 10
2 jack 11
3 jerry 12
`)
const codeHistory = ref<string[]>([])
const clipboardError = ref('')
const activeTab = ref('clipboard')
const inputDelimiter = ref('\t')
const inputDelimiterDisplay = ref('\\t')
const tableName = ref('tb1')
const isLoading = ref(false)

const csvText = ref('')
const csvHeaders = ref<string[]>([])
const selectedHeaders = ref<string[]>([])

// 監聽顯示值的變化並更新實際值
watch(inputDelimiterDisplay, (newValue) => {
  // 將顯示的轉義字符轉換為實際的分隔符
  switch (newValue) {
    case '\\t':
      inputDelimiter.value = '\t'
      break
    case '\\n':
      inputDelimiter.value = '\n'
      break
    case '\\r':
      inputDelimiter.value = '\r'
      break
    default:
      inputDelimiter.value = newValue
  }
})

// 當 selectedHeaders 變化時，根據勾選的欄位裁切 csvText 並更新 code
watch(selectedHeaders, (newHeaders) => {
  if (csvText.value && newHeaders.length > 0) {
    // 依照 csvHeaders 的順序排列 newHeaders
    const orderedHeaders = csvHeaders.value.filter(h => newHeaders.includes(h))
    code.value = cutCsvText(csvText.value, orderedHeaders, ',')
  }
})

// 新增：標籤頁列表
interface TabItem {
  id: string
  name: string
}

const tabList: TabItem[] = [
  { id: 'clipboard', name: '剪貼簿' },
  { id: 'csv', name: 'CSV' },
  { id: 'translate', name: '翻譯' },
]

function clickTableToCsv() {
  const inputText = code.value
  code.value = convertTableFormatToCsv(inputText, inputDelimiter.value);
  if (code.value !== inputText) {
    pasteToClipboard(code.value);
  }
  csvText.value = code.value
  csvHeaders.value = getCsvHeadersName(csvText.value, ',')
  selectedHeaders.value = csvHeaders.value
}

function clickJsonToCsv() {
  const inputText = code.value
  code.value = convertJsonFormatToCsv(inputText);
  if (code.value !== inputText) {
    pasteToClipboard(code.value);
  }
}

function clickCsvToJson() {
  const inputText = code.value
  code.value = convertCsvFormatToJson(inputText);
  if (code.value !== inputText) {
    pasteToClipboard(code.value);
  }
}

function clickCsvToTable() {
  const inputText = code.value
  code.value = convertCsvFormatToTable(inputText, inputDelimiter.value);
  if (code.value !== inputText) {
    pasteToClipboard(code.value);
  }
}

function clickCsvToSql() {
  const inputText = code.value
  code.value = convertCsvFormatToSql(inputText, tableName.value);
  if (code.value !== inputText) {
    pasteToClipboard(code.value);
  }
}


async function clickTranslateToEn() {
  isLoading.value = true;
  const inputText = code.value;
  const result = await translateToEnAsync(inputText);
  code.value = result;
  if (code.value !== inputText) {
    pasteToClipboard(code.value);
  }
  isLoading.value = false;
}

async function clickTranslateToZh() {
  isLoading.value = true;
  const inputText = code.value;
  const result = await translateToZhAsync(inputText);
  code.value = result;
  if (code.value !== inputText) {
    pasteToClipboard(code.value);
  }
  isLoading.value = false;
}

async function handleCopyFromClipboard() {
  await copyFromClipboard(
    (text) => {
      code.value = text;
      clipboardError.value = '';
    },
    (error) => {
      clipboardError.value = error;
    }
  );
}

function clickUndo() {
  code.value = codeHistory.value.shift() || ''
}

function handleCodeChange(event: Event) {
  const target = event.target as HTMLTextAreaElement
  code.value = target.value
  // 記錄到 codeHistory，只保留最近 10 筆
  codeHistory.value.unshift(code.value)
  if (codeHistory.value.length > 10) {
    codeHistory.value.length = 10
  }
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
  <Loading v-if="isLoading" />
  <div class="control-panel">
    <div class="tabs">
      <button 
        v-for="tab in tabList"
        :key="tab.id"
        class="tab-button" 
        :class="{ active: activeTab === tab.id }"
        @click="activeTab = tab.id"
      >
        {{ tab.name }}
      </button>
    </div>

    <div class="tab-panels">
      <!-- 剪貼簿面板 -->
      <div v-if="activeTab === 'clipboard'" class="tab-panel">
        <button type="button" @click="handleCopyFromClipboard">從剪貼簿複製</button>
        <p v-if="clipboardError" class="error-message">{{ clipboardError }}</p>
      </div>

      <!-- CSV 面板 -->
      <div v-if="activeTab === 'csv'" class="tab-panel">
        <div class="buttons-row">
          <button type="button" @click="clickTableToCsv">Table To CSV</button>
          <button type="button" @click="clickJsonToCsv">Json To CSV</button>
          <button type="button" @click="clickCsvToJson">CSV To Json</button>
          <button type="button" @click="clickCsvToTable">CSV To Table</button>
          <button type="button" @click="clickCsvToSql">CSV To SQL</button>
        </div>
        <div class="delimiter-row">
          <label for="delimiter-input">Table delimiter:</label>
          <input 
            id="delimiter-input"
            type="text" 
            v-model="inputDelimiterDisplay"
            class="delimiter-input"
            title="使用 \t 表示 Tab，\n 表示換行"
          />
        </div>
        <div class="delimiter-row">
          <label for="delimiter-input">To Table Name:</label>
          <input 
            type="text" 
            v-model="tableName"
            class="delimiter-input"
            title="to table name"
          />
        </div>

        <!-- 顯示表頭（checkbox 方式，每個 header 一個 checkbox） -->
        <div class="delimiter-row" style="flex-direction: column; align-items: flex-start;">
          <label>Table headers:</label>
          <div style="display: flex; flex-direction: row; flex-wrap: wrap; gap: 10px;">
            <div v-for="header in csvHeaders" :key="header">
              <label>
                <input type="checkbox" :value="header" v-model="selectedHeaders" />
                {{ header }}
              </label>
            </div>
          </div>
        </div>
        
        <div class="buttons-row">
          <button type="button" @click="clickUndo">Undo</button>
        </div>
      </div>

      <!-- 翻譯面板 -->
      <div v-if="activeTab === 'translate'" class="tab-panel">
        <div class="buttons-row">
          <button type="button" @click="clickTranslateToEn">To 英文</button>
          <button type="button" @click="clickTranslateToZh">To 中文</button>
        </div>
      </div>
    </div>
  </div>

  <div class="editor-container">
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
</template>

<style scoped>
.read-the-docs {
  color: #888;
}

.editor-container {
  margin: 20px auto;
  width: 90%;
  min-width: 400px;
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
  border: 1px solid #333;
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
  margin: 0;
  padding: 8px;
  color: #ff6b6b;
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

.control-panel {
  margin: 20px auto;
  width: 90%;
  min-width: 400px;
  border: 1px solid #333;
  border-radius: 4px;
  overflow: hidden;
  background-color: #1e1e1e;
}

.tabs {
  display: flex;
  background-color: #252526;
  border-bottom: 1px solid #333;
}

.tab-button {
  padding: 10px 20px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 14px;
  color: #969696;
  transition: all 0.3s ease;
  border-right: 1px solid #333;
}

.tab-button:hover {
  background-color: #2d2d2d;
  color: #ffffff;
}

.tab-button.active {
  background-color: #1e1e1e;
  color: #4CAF50;
  border-bottom: 2px solid #4CAF50;
}

.tab-panels {
  background-color: #1e1e1e;
}

.tab-panel {
  padding: 15px;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.buttons-row {
  display: flex;
  flex-direction: row;
  gap: 10px;
  align-items: center;
}

.delimiter-row {
  display: flex;
  flex-direction: row;
  gap: 10px;
  align-items: center;
}

.delimiter-row label {
  color: #d4d4d4;
  font-size: 14px;
}

.delimiter-input {
  background-color: #1e1e1e;
  border: 1px solid #333;
  color: #d4d4d4;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 14px;
  width: 60px;
  transition: border-color 0.3s ease;
}

.delimiter-input:focus {
  outline: none;
  border-color: #4CAF50;
}

.delimiter-input:hover {
  border-color: #666;
}

.buttons-row button {
  background-color: #4CAF50;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.3s ease;
  white-space: nowrap;
}

.buttons-row button:hover {
  background-color: #45a049;
}

.control-panel {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.tab-button {
  position: relative;
  overflow: hidden;
}

.tab-button::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: #4CAF50;
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.tab-button.active::after {
  transform: scaleX(1);
}

.code-editor {
  border-color: #333;
  background-color: #1e1e1e;
  color: #d4d4d4;
}
</style>
