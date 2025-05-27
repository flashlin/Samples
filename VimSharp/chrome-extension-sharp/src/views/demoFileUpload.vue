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
  <div class="min-h-screen flex items-center justify-center bg-gray-900 w-full">
    <div class="flex flex-col items-center w-full">
      <h1 class="text-2xl font-bold mb-4 text-white text-center">Demo</h1>
      <div class="w-full max-w-md mx-auto bg-white border border-gray-700 shadow-lg rounded-xl p-6">
        <FileUpload :fileList="fileList" />
      </div>
    </div>
  </div>
</template> 