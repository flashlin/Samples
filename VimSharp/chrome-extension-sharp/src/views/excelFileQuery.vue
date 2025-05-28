<script lang="ts" setup>
import FileUpload, { FileUploadInstance } from '@/components/FileUpload.vue';
import { ExcelSheet, getExcelFileAsync } from '@/tools/excel';
import { ref, onMounted, onUnmounted } from 'vue';
import VimCodeEditor from '@/components/vimCodeEditor.vue';
import DataTable from '@/components/DataTable.vue';

interface ExcelFile {
  fileName: string;
  sheets: ExcelSheet[];
}

const excelFiles = ref<ExcelFile[]>([]);
const code = ref('from tb1 in test select tb1')

// 動態同步左右高度
const leftRef = ref<HTMLElement | null>(null)
const rightHeight = ref('auto')

function syncRightHeight() {
  if (leftRef.value) {
    rightHeight.value = leftRef.value.clientHeight + 'px'
  }
  console.log("🚀 ~ syncRightHeight ~ rightHeight.value:", rightHeight.value)
}

onMounted(() => {
  syncRightHeight()
  window.addEventListener('resize', syncRightHeight)
})
onUnmounted(() => {
  window.removeEventListener('resize', syncRightHeight)
})

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
    }
  }catch{
    //
  }
  if (instance.processBarStatus === initialStatus) {
    instance.processBarStatus = 'Uploaded!';
  }
  console.log("call syncRightHeight")
  syncRightHeight()
}
</script>

<template>
  <div class="min-h-screen flex flex-col items-center justify-center bg-gray-900 w-full">
    <div class="flex flex-row items-stretch w-full max-w-7xl min-h-[600px]">
      <!-- 左側：原本內容 -->
      <div ref="leftRef" class="flex flex-col items-center w-1/2 pr-4 h-full">
        <h1 class="text-2xl font-bold mb-4 text-white text-center">Excel Query</h1>
        <div class="w-full mx-auto border border-gray-700 shadow-lg rounded-xl p-6 flex justify-center"
          style="background:#2d333b;">
          <FileUpload accept=".xlsx,.csv" :processHandler="uploadAllExcelFiles" processButtonTitle="Import"
            style="width:98%;" />
        </div>
      </div>
      <!-- Layout 右側 -->
      <div class="flex flex-col w-1/2 pl-4 gap-6 justify-start overflow-y-auto" :style="{height: rightHeight}">
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
    </div>
    <!-- layout 下方 -->
    <div class="w-full h-96 mx-auto border border-gray-700 shadow-lg rounded-xl p-6 flex justify-center mt-6"
      style="background:#23272f;">
      <VimCodeEditor v-model:value="code" :enableVim="false" class="w-full h-full" />
    </div>
  </div>
</template>