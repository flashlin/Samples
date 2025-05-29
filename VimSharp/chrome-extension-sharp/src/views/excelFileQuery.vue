<script lang="ts" setup>
import FileUpload, { FileUploadInstance } from '@/components/FileUpload.vue';
import { convertSheetToDataTable, ExcelSheet, getExcelFileAsync } from '@/tools/excel';
import { ref } from 'vue';
import VimCodeEditor from '@/components/vimCodeEditor.vue';
import DataTable from '@/components/DataTable.vue';
import { hello } from '@/tools/waSqlite';
import { DataTable as DataTableType } from '@/tools/dataTypes';

interface ExcelFile {
  fileName: string;
  sheets: ExcelSheet[];
}

const excelFiles = ref<ExcelFile[]>([]);
const allDataTables = ref<DataTableType[]>([]);
const code = ref('from tb1 in test select tb1')

async function uploadAllExcelFiles(files: File[], instance: FileUploadInstance) {
  const initialStatus = 'Uploading...';
  instance.processBarStatus = initialStatus;
  try{
    for (const file of files) {
      const excelSheets = await getExcelFileAsync(file);
      excelFiles.value.push({
        fileName: file.name,
        sheets: excelSheets
      });
      const dataTables = excelSheets.map(convertSheetToDataTable);
      allDataTables.value.push(...dataTables);
    }
  }catch{
    //
  }
  if (instance.processBarStatus === initialStatus) {
    instance.processBarStatus = 'Uploaded!';
  }
}

async function sayHello() {
  await hello()
}
</script>


<template>
  <div class="min-h-screen flex flex-col items-center justify-center bg-gray-900 w-full">
    <button @click="sayHello">hello</button> 
    <!-- Header -->
    <div ref="leftRef" class="w-full max-w-7xl border border-gray-700 shadow-lg rounded-xl p-6 flex flex-col items-center mb-4" style="background:#2d333b;">
      <h1 class="text-2xl font-bold mb-4 text-white text-center">Excel Query</h1>
      <FileUpload accept=".xlsx,.csv" :processHandler="uploadAllExcelFiles" processButtonTitle="Import" style="width:98%;" />
    </div>
    <div class="flex flex-row w-full max-w-7xl min-h-[600px] flex-1">
      <!-- Sidebar -->
      <div class="flex flex-col w-1/3 pr-4 gap-6 justify-start overflow-y-auto">
        <div class="flex-1">
          <DataTable />
        </div>
        <div class="flex-1">
          <DataTable />
        </div>
        <div class="flex-1">
          <DataTable />
        </div>
      </div>
      <!-- Main Content -->
      <div class="flex-1 flex flex-col w-2/3 pl-4">
        <div class="w-full h-96 border border-gray-700 shadow-lg rounded-xl p-6 flex justify-center mt-0" style="background:#23272f;">
          <VimCodeEditor v-model:value="code" :enableVim="false" class="w-full h-full" />
        </div>
      </div>
    </div>
  </div>
</template>