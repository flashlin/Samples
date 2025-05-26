<script lang="ts" setup>
import FileUpload from '../components/FileUpload.vue';

// Sample file list for demonstration
const fileList = [
    { name: 'document1.pdf' },
    { name: 'image1.jpg' },
    { name: 'video1.mp4' }
];
</script>

<template>
    <div class="container mx-auto p-4">
        <h1 class="text-2xl font-bold mb-4">檔案上傳示範</h1>
        
        <!-- File Upload Component -->
        <FileUpload :fileList="fileList" />
    </div>
</template> 