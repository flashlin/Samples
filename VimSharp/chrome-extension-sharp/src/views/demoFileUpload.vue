<script lang="ts" setup>
import FileUpload from '@/components/FileUpload.vue';

// Sample file list for demonstration
const fileList = [
    { name: 'document1.pdf' },
    { name: 'image1.jpg' },
    { name: 'video1.mp4' }
];
</script>

<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-900">
    <div>
      <h1 class="text-2xl font-bold mb-4 text-white text-center">檔案上傳示範</h1>
      <FileUpload :fileList="fileList" />
    </div>
  </div>
</template> 