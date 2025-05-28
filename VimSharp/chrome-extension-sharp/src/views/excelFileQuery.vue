<script lang="ts" setup>
import FileUpload, { FileUploadInstance } from '@/components/FileUpload.vue';


async function uploadAllExcelFiles(_files: File[], instance: FileUploadInstance) {
  const initialStatus = 'Uploading...';
  instance.processBarStatus = initialStatus;
  //for (const file of files) {
    //const arrayBuffer = await file.arrayBuffer();
  //}
  if (instance.processBarStatus === initialStatus) {
    instance.processBarStatus = 'Uploaded!';
  }
}
</script>

<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-900 w-full">
    <div class="flex flex-col items-center w-full">
      <h1 class="text-2xl font-bold mb-4 text-white text-center">Demo</h1>
      <div class="w-full mx-auto border border-gray-700 shadow-lg rounded-xl p-6 flex justify-center" style="background:#2d333b;">
        <FileUpload accept=".jpg,.txt" :processHandler="uploadAllExcelFiles" 
          processButtonTitle="Upload" style="width:98%;" />
      </div>
    </div>
  </div>
</template> 