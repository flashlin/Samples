<script setup lang="ts">
import { ref, watch } from 'vue'
import { getCsvDataTableColumns, processCsvRowsToString } from '../tools/textTool'
import Loading from '@/components/Loading.vue'
import CodeEditor from '@/components/CodeEditor.vue';
import Handlebars from 'handlebars';

const generateTemplateEditorRef = ref<any>(null)
const concatTemplate = ref('')
const resultEditorRef = ref<any>(null)

const csvText = ref(`id\tname\tage
1\tflash\t10
2\tjack\t11
3\tjerry\t12
`)
const generateTemplate = ref(``)
const result = ref(``)
const inputDelimiter = ref('\t')
const inputDelimiterDisplay = ref('\\t')
const isLoading = ref(false)

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

function clickGenerateCsvToJsonTemplate() {
  const csvDataColumns = getCsvDataTableColumns(csvText.value, inputDelimiter.value)

  // 轉換 headers 為 "${header}": {{${header}}}
  const templateBody = csvDataColumns.map(header => {
    const value = header.type === 'TEXT' ? `"{{${header.name}}}"` : `{{${header.name}}}`;
    return `  "${header.name}": ${value}`;
  }).join(",\n");
  const jsonTemplate = `{
${templateBody}
}`;

  generateTemplate.value = jsonTemplate;
}

function generate() {
  const template = Handlebars.compile(generateTemplate.value);
  const concatDelimiter = concatTemplate.value;
  let first = true;
  const output = processCsvRowsToString(csvText.value, inputDelimiter.value, (row: any, headers: string[]) => {
    const rowData: Record<string, any> = {};
    headers.forEach(header => {
      rowData[header] = row[header];
    });
    let concatDelimiterString = first ? '' : concatDelimiter;
    const rowString = concatDelimiterString +template(rowData);
    first = false;
    return rowString;
  });
  result.value = output;
}
</script>

<template>
  <Loading v-if="isLoading" />
  <div class="editor-container" style="display: flex; flex-direction: column; gap: 10px;">

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


    Csv Content:
    <textarea 
      class="code-editor" 
      v-model="csvText" 
      spellcheck="false"
    ></textarea>

    Generate template:
    <button @click="clickGenerateCsvToJsonTemplate">Json Template</button>
    <div style="height: 200px;">
      <CodeEditor ref="generateTemplateEditorRef" v-model="generateTemplate" class="w-full h-full" />
    </div>
    Concat template:
    <textarea class="code-editor" v-model="concatTemplate" ></textarea>
    <div class="editor-actions">
      <button @click="generate" class="run-button">Generate</button>
    </div>

    Result:
    <div style="height: 300px;">
      <CodeEditor ref="resultEditorRef" v-model="result" class="w-full h-full" />
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
  height: 200px;
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
    height: 200px;
  }
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

.code-editor {
  border-color: #333;
  background-color: #1e1e1e;
  color: #d4d4d4;
}
</style>
