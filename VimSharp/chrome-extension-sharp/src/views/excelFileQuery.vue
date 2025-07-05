<script lang="ts" setup>
import FileUpload, { FileUploadInstance } from '@/components/FileUpload.vue';
import { convertSheetToDataTable, ExcelSheet, getExcelFileAsync } from '@/tools/excelKit';
import { ref, onMounted, onUnmounted } from 'vue';
//import VimCodeEditor from '@/components/vimCodeEditor.vue';
import VimCodeEditor from '@/components/CodeEditor.vue';
import { createTableAsync, dropTableAsync, insertDataTableAsync, querySqliteAsync } from '@/tools/waSupport';
import { DataTable as DataTableType } from '@/tools/dataTypes';
import LargeDataTable from '@/components/LargeDataTable.vue';
import { goTo } from '@/tools/visual-router'
import { useSupportStore } from '@/SupportStore';
import type { IntellisenseContext } from '@/components/CodeEditorTypes';
import { useIntellisenseApi } from '@/tools/intellisenseApi';

interface ExcelFile {
  fileName: string;
  sheets: ExcelSheet[];
}

const excelFiles = ref<ExcelFile[]>([]);
const allDataTables = ref<DataTableType[]>([]);
const code = ref('')
const errorMessage = ref('');
const supportStore = useSupportStore();
const vimEditorRef = ref<any>(null)
const intellisenseApi = useIntellisenseApi();

async function uploadAllExcelFiles(files: File[], instance: FileUploadInstance) {
  const initialStatus = 'Uploading...';
  instance.processBarStatus = initialStatus;
  let newDataTables: DataTableType[] = [];
  try{
    for (const file of files) {
      const excelSheets = await getExcelFileAsync(file);
      excelFiles.value.push({
        fileName: file.name,
        sheets: excelSheets
      });
      const dataTables = excelSheets.map(convertSheetToDataTable);
      newDataTables.push(...dataTables);
    }
  }catch(e){
    console.error(e);
  }
  
  // Check for duplicate tableNames before pushing to allDataTables
  for (const newTable of newDataTables) {
    const isDuplicate = allDataTables.value.some(existingTable => existingTable.tableName === newTable.tableName);
    if (!isDuplicate) {
      allDataTables.value.push(newTable);
      await dropTableAsync(newTable.tableName);
      await createTableAsync(newTable);
      await insertDataTableAsync(newTable, newTable.tableName);
    }
  }

  if (instance.processBarStatus === initialStatus) {
    instance.processBarStatus = 'Uploaded!';
    //await backupSqliteDbAsync();
  }
}

async function executeQuery() {
  try {
    const result = await querySqliteAsync(code.value, {})
    supportStore.setQueryResult(result);
    errorMessage.value = '';
  } catch (e) {
    errorMessage.value = e as string;
  }
}

async function deleteTables() {
  for (const dt of allDataTables.value) {
    await dropTableAsync(dt.tableName);
  }
  allDataTables.value = [];
}

function test() {
  vimEditorRef.value?.showIntellisense([
    { title: 'a', context: 'abc123' },
    { title: 'b', context: 'You are winner' }
  ])
}

async function handleMyIntellisense(context: IntellisenseContext): Promise<any[]> {
  const question = `${context.content[0]}{{cursor}}${context.content[1]}`;
  try {
    const resp = await intellisenseApi.getIntellisenseList({ question });
    const result = resp.items.map(item => ({
      title: `${item.confidence_score} ${item.context.substring(0, 30)}`,
      context: item.context
    }));
    if( result.length == 0 )
    {
      return [
        { title: '<No result>', context: '' }
      ]
    }
    return result;
  } catch (e) {
    return [];
  }
}

function handleF8Key(e: KeyboardEvent) {
  // F8 對應 key 為 'F8'
  if (e.key === 'F8') {
    executeQuery();
  }
}

onMounted(async () => {
  // F8 快捷鍵監聽
  window.addEventListener('keydown', handleF8Key);
  //await restoreSqliteDbAsync();
});
onUnmounted(() => {
  window.removeEventListener('keydown', handleF8Key);
});
</script>


<template>
  <div class="items-center justify-center bg-gray-900 w-full">
    <!-- Home Button -->
    <div class="fixed top-4 left-4 z-50">
      <button
        @click="goTo('')"
        class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded shadow-lg border border-blue-800"
        style="min-width: 80px;"
      >
        Home
      </button>
    </div>
    <!-- Header -->
    <div class="w-full max-w-7xl border border-gray-700 shadow-lg rounded-xl p-6 flex flex-col items-center mb-4" style="background:#2d333b;">
      <h1 class="text-2xl font-bold mb-4 text-white text-center">Excel Query</h1>
      <FileUpload accept=".xlsx,.csv" :processHandler="uploadAllExcelFiles" processButtonTitle="Import" style="width:98%;" />
    </div>
    <!-- Sidebar -->
    <div class="w-full max-w-7xl flex-col" style="background:#23272f;">
      <div class="flex-1" v-for="(dt, idx) in allDataTables" :key="dt.tableName + idx">
        <LargeDataTable :dt="dt" />
      </div>
    </div>
    <!-- Main Content -->
    <div class="w-full max-w-7xl flex flex-col gap-4">
      <div class="w-full h-96 border border-gray-700 shadow-lg rounded-xl p-6 flex flex-col justify-center mt-0" style="background:#23272f;">
        <div class="flex flex-row gap-2 mb-2">
          <button @click="executeQuery">Execute (F8)</button>
          <button @click="deleteTables">Delete (F4)</button>
          <button @click="test">Test Intellisense</button>
          <!-- 這裡未來可放更多按鈕 -->
        </div>
        <VimCodeEditor ref="vimEditorRef" v-model="code" :enableVim="false" class="w-full h-full" :onShowIntellisense="handleMyIntellisense" />
      </div>
      <div v-if="errorMessage" class="w-full flex flex-col justify-center mt-0" style="background:#23272f; min-height: 48px;">
        <p class="text-red-500">{{ errorMessage }}</p>
      </div>
      <div class="w-full">
        <LargeDataTable :dt="supportStore.queryResult" />
      </div>
    </div>
    <!-- 保留底部三行空白 -->
    <div class="footer-blank"></div>
  </div>
</template>

<style scoped>
.footer-blank {
  min-height: 3em; /* 3行字的高度，可依需求調整 */
}
</style>