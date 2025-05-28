<script lang="ts" setup>
import FileUpload from '@/components/FileUpload.vue';
import { UploadFileApi, UploadFileRequest } from '@/tools/uploadFileApi';

const apiUrl = import.meta.env.VITE_API_URL;
const uploader = new UploadFileApi(apiUrl);

async function uploadAllFiles(files: File[]) {
  for (const file of files) {
    const arrayBuffer = await file.arrayBuffer();
    const req: UploadFileRequest = {
      fileName: file.name,
      fileContent: new Uint8Array(arrayBuffer),
      offset: 0
    };
    try {
      const resp = await uploader.upload(req);
      console.log('Uploaded:', resp.fileName);
    } catch (err) {
      console.error('Upload failed:', file.name, err);
    }
  }
  alert('All files uploaded!');
}
</script>

<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-900 w-full">
    <div class="flex flex-col items-center w-full">
      <h1 class="text-2xl font-bold mb-4 text-white text-center">Demo</h1>
      <div class="w-full mx-auto border border-gray-700 shadow-lg rounded-xl p-6 flex justify-center" style="background:#2d333b;">
        <FileUpload accept=".jpg,.txt" :processHandler="uploadAllFiles" style="width:98%;" />
      </div>
    </div>
  </div>
</template> 