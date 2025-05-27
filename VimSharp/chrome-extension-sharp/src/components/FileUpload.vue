<script lang="ts" setup>
import { ref } from 'vue';

interface FileProgressItem {
    name: string;
    size: number;
    progress: number;
    status: 'uploading' | 'success' | 'error';
}
interface FileItem {
    name: string;
}
interface FileUploadProps {
    fileList: FileItem[];
}

const props = defineProps<FileUploadProps>();
const fileProgressList = ref<FileProgressItem[]>([
    {
        name: 'preline-ui.html',
        size: 7 * 1024, // 7 KB
        progress: 1,
        status: 'uploading'
    },
    {
        name: 'preline-ui.mp4',
        size: 105.5 * 1024 * 1024, // 105.5 MB
        progress: 1,
        status: 'uploading'
    },
    {
        name: 'preline-ui-cover.jpg',
        size: 55 * 1024, // 55 KB
        progress: 1,
        status: 'uploading'
    }
]);

// Format file size to human readable format
const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

// Remove file by index
const removeFile = (idx: number) => {
    fileProgressList.value.splice(idx, 1);
};
</script>

<template>
  <!-- File Uploading Progress Form -->
  <div class="flex flex-col bg-[#181c20] border border-[#23272f] shadow-2xs rounded-xl">
    <!-- Body -->
    <div class="p-2 space-y-2">
      <div v-for="(item, idx) in fileProgressList" :key="item.name" class="bg-gray-800 rounded-lg px-4 py-3 flex flex-col gap-1">
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-x-3">
            <span class="size-7 flex justify-center items-center border border-[#23272f] text-white rounded-lg">
              <svg class="shrink-0 size-5" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="white"/>
                <polyline points="17 8 12 3 7 8" stroke="white"/>
                <line x1="12" x2="12" y1="3" y2="15" stroke="white"/>
              </svg>
            </span>
            <div>
              <p class="text-sm font-medium text-white">{{ item.name }}</p>
              <p class="text-xs text-neutral-300">{{ formatFileSize(item.size) }}</p>
            </div>
          </div>
          <div class="inline-flex items-center gap-x-2">
            <button type="button" class="text-neutral-400 hover:text-white">
              <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect width="4" height="16" x="6" y="4" stroke="currentColor"/>
                <rect width="4" height="16" x="14" y="4" stroke="currentColor"/>
              </svg>
              <span class="sr-only">Pause</span>
            </button>
            <button type="button" @click="removeFile(idx)" class="text-neutral-400 hover:text-red-400">
              <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 6h18" stroke="currentColor"/>
                <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" stroke="currentColor"/>
                <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" stroke="currentColor"/>
                <line x1="10" x2="10" y1="11" y2="17" stroke="currentColor"/>
                <line x1="14" x2="14" y1="11" y2="17" stroke="currentColor"/>
              </svg>
              <span class="sr-only">Delete</span>
            </button>
          </div>
        </div>
        <!-- Progress Bar -->
        <div class="w-full h-1 bg-transparent mt-1">
          <div class="h-1 bg-blue-500 rounded-full" :style="{ width: (item.progress * 100) + '%' }"></div>
        </div>
      </div>
    </div>
    <!-- End Body -->
    <!-- Footer -->
    <div class="bg-gray-800 border-t border-[#23272f] rounded-b-xl py-2 px-4 flex justify-between items-center">
      <span class="text-sm font-semibold text-white">{{ fileProgressList.length }} left</span>
      <div class="flex items-center gap-x-2">
        <button type="button" class="text-neutral-400 hover:text-white">
          <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="4" height="16" x="6" y="4" stroke="currentColor"/>
            <rect width="4" height="16" x="14" y="4" stroke="currentColor"/>
          </svg>
          <span class="sr-only">Pause</span>
        </button>
        <button type="button" class="text-neutral-400 hover:text-red-400 flex items-center gap-x-1">
          <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 6h18" stroke="currentColor"/>
            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" stroke="currentColor"/>
            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" stroke="currentColor"/>
            <line x1="10" x2="10" y1="11" y2="17" stroke="currentColor"/>
            <line x1="14" x2="14" y1="11" y2="17" stroke="currentColor"/>
          </svg>
          <span>Delete</span>
        </button>
      </div>
    </div>
    <!-- End Footer -->
  </div>
  <!-- End File Uploading Progress Form -->
</template>
