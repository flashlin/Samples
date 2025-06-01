<script lang="ts" setup>
import { ref, computed } from 'vue';

export interface FileUploadInstance {
  processBarStatus: string;
}

interface FileUploadProps {
    accept?: string;
    processHandler?: (files: File[], instance: FileUploadInstance) => void | Promise<void>;
    processButtonTitle?: string;
}

const props = withDefaults(defineProps<FileUploadProps>(), {
  processButtonTitle: 'Process',
});

const fileList = ref<File[]>([]);
const processBarStatus = ref('');
const instance: FileUploadInstance = {
  get processBarStatus() {
    return processBarStatus.value;
  },
  set processBarStatus(val: string) {
    processBarStatus.value = val;
  }
};

function onFileChange(e: Event) {
  const input = e.target as HTMLInputElement;
  if (input.files) {
    for (let i = 0; i < input.files.length; i++) {
      fileList.value.push(input.files[i]);
    }
  }
  input.value = '';
}

const fileProgressList = computed(() =>
    fileList.value.map(file => ({
        name: file.name,
        size: file.size,
        progress: 1,
        status: 'uploading' as const
    }))
);

// Format file size to human readable format
const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

function handleAllFileProcess() {
  if (props.processHandler) {
    props.processHandler(fileList.value, instance);
  }
}

function removeFile(index: number) {
  // Remove file from fileList by index
  fileList.value.splice(index, 1);
}
</script>

<template>
  <!-- File Uploading Progress Form -->
  <div class="flex flex-col bg-[#181c20] border border-[#23272f] shadow-2xs rounded-xl">
    <!-- 上傳按鈕與 input -->
    <div class="flex justify-end items-center mb-4 mt-4 mr-4">
      <label
        class="flex bg-gray-800 hover:bg-gray-700 text-white text-base font-medium px-4 py-2.5 outline-none rounded w-max cursor-pointer mx-0">
        <svg xmlns="http://www.w3.org/2000/svg" class="w-6 mr-2 fill-white inline" viewBox="0 0 32 32">
          <path
            d="M23.75 11.044a7.99 7.99 0 0 0-15.5-.009A8 8 0 0 0 9 27h3a1 1 0 0 0 0-2H9a6 6 0 0 1-.035-12 1.038 1.038 0 0 0 1.1-.854 5.991 5.991 0 0 1 11.862 0A1.08 1.08 0 0 0 23 13a6 6 0 0 1 0 12h-3a1 1 0 0 0 0 2h3a8 8 0 0 0 .75-15.956z"
            data-original="#000000" />
          <path
            d="M20.293 19.707a1 1 0 0 0 1.414-1.414l-5-5a1 1 0 0 0-1.414 0l-5 5a1 1 0 0 0 1.414 1.414L15 16.414V29a1 1 0 0 0 2 0V16.414z"
            data-original="#000000" />
        </svg>
        Select Files
        <input type="file" class="hidden" @change="onFileChange" multiple :accept="props.accept" />
      </label>
    </div>
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
            <button type="button" class="text-neutral-400 hover:text-red-400" @click="removeFile(idx)">
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
      <!-- Process Status -->
      <div v-if="processBarStatus" class="text-xs text-red-400 mt-1">{{ processBarStatus }}</div>
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
        <button @click="handleAllFileProcess" class="ml-4 bg-blue-600 hover:bg-blue-700 text-white font-medium px-4 py-2.5 rounded" :disabled="fileList.length === 0">
          {{ props.processButtonTitle }}
        </button>
      </div>
    </div>
    <!-- End Footer -->
  </div>
  <!-- End File Uploading Progress Form -->
</template>
