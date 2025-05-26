<script lang="ts" setup>
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
const fileProgressList: FileProgressItem[] = [];

</script>

<template>
<!-- File Uploading Progress Form -->
<div class="flex flex-col bg-white border border-gray-200 shadow-2xs rounded-xl">
  <!-- Body -->
  <div class="p-4 md:p-5 space-y-7">
    <div>
      <!-- Uploading File Content -->
      <div class="mb-2 flex justify-between items-center">
        <div class="flex items-center gap-x-3">
          <span class="size-8 flex justify-center items-center border border-gray-200 text-gray-500 rounded-lg">
            <svg class="shrink-0 size-5" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" x2="12" y1="3" y2="15"></line>
            </svg>
          </span>
          <div>
            <p class="text-sm font-medium text-gray-800">preline-ui.html</p>
            <p class="text-xs text-gray-500">7 KB</p>
          </div>
        </div>
        <div class="inline-flex items-center gap-x-2">
          <button type="button" class="relative text-gray-500 hover:text-gray-800 focus:outline-hidden focus:text-gray-800 disabled:opacity-50 disabled:pointer-events-none">
            <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect width="4" height="16" x="6" y="4"></rect>
              <rect width="4" height="16" x="14" y="4"></rect>
            </svg>
            <span class="sr-only">Pause</span>
          </button>
          <button type="button" class="relative text-gray-500 hover:text-gray-800 focus:outline-hidden focus:text-gray-800 disabled:opacity-50 disabled:pointer-events-none">
            <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 6h18"></path>
              <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
              <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
              <line x1="10" x2="10" y1="11" y2="17"></line>
              <line x1="14" x2="14" y1="11" y2="17"></line>
            </svg>
            <span class="sr-only">Delete</span>
          </button>
        </div>
      </div>
      <!-- End Uploading File Content -->

      <!-- Progress Bar -->
      <div class="flex w-full h-2 bg-gray-200 rounded-full overflow-hidden" role="progressbar" aria-valuenow="1" aria-valuemin="0" aria-valuemax="100">
        <div class="flex flex-col justify-center rounded-full overflow-hidden bg-blue-600 text-xs text-white text-center whitespace-nowrap transition duration-500" style="width: 1%"></div>
      </div>
      <!-- End Progress Bar -->
    </div>

    <div>
      <!-- Uploading File Content -->
      <div class="mb-2 flex justify-between items-center">
        <div class="flex items-center gap-x-3">
          <span class="size-8 flex justify-center items-center border border-gray-200 text-gray-500 rounded-lg">
            <svg class="shrink-0 size-5" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" x2="12" y1="3" y2="15"></line>
            </svg>
          </span>
          <div>
            <p class="text-sm font-medium text-gray-800">preline-ui.mp4</p>
            <p class="text-xs text-gray-500">105.5 MB</p>
          </div>
        </div>
        <div class="inline-flex items-center gap-x-2">
          <button type="button" class="relative text-gray-500 hover:text-gray-800 focus:outline-hidden focus:text-gray-800 disabled:opacity-50 disabled:pointer-events-none">
            <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect width="4" height="16" x="6" y="4"></rect>
              <rect width="4" height="16" x="14" y="4"></rect>
            </svg>
            <span class="sr-only">Pause</span>
          </button>
          <button type="button" class="relative text-gray-500 hover:text-gray-800 focus:outline-hidden focus:text-gray-800 disabled:opacity-50 disabled:pointer-events-none">
            <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 6h18"></path>
              <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
              <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
              <line x1="10" x2="10" y1="11" y2="17"></line>
              <line x1="14" x2="14" y1="11" y2="17"></line>
            </svg>
            <span class="sr-only">Delete</span>
          </button>
        </div>
      </div>
      <!-- End Uploading File Content -->

      <!-- Progress Bar -->
      <div class="flex w-full h-2 bg-gray-200 rounded-full overflow-hidden" role="progressbar" aria-valuenow="1" aria-valuemin="0" aria-valuemax="100">
        <div class="flex flex-col justify-center rounded-full overflow-hidden bg-blue-600 text-xs text-white text-center whitespace-nowrap transition duration-500" style="width: 1%"></div>
      </div>
      <!-- End Progress Bar -->
    </div>

    <div>
      <!-- Uploading File Content -->
      <div class="mb-2 flex justify-between items-center">
        <div class="flex items-center gap-x-3">
          <span class="size-8 flex justify-center items-center border border-gray-200 text-gray-500 rounded-lg">
            <svg class="shrink-0 size-5" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" x2="12" y1="3" y2="15"></line>
            </svg>
          </span>
          <div>
            <p class="text-sm font-medium text-gray-800">preline-ui-cover.jpg</p>
            <p class="text-xs text-gray-500">55 KB</p>
          </div>
        </div>
        <div class="inline-flex items-center gap-x-2">
          <button type="button" class="relative text-gray-500 hover:text-gray-800 focus:outline-hidden focus:text-gray-800 disabled:opacity-50 disabled:pointer-events-none">
            <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect width="4" height="16" x="6" y="4"></rect>
              <rect width="4" height="16" x="14" y="4"></rect>
            </svg>
            <span class="sr-only">Pause</span>
          </button>
          <button type="button" class="relative text-gray-500 hover:text-gray-800 focus:outline-hidden focus:text-gray-800 disabled:opacity-50 disabled:pointer-events-none">
            <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 6h18"></path>
              <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
              <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
              <line x1="10" x2="10" y1="11" y2="17"></line>
              <line x1="14" x2="14" y1="11" y2="17"></line>
            </svg>
            <span class="sr-only">Delete</span>
          </button>
        </div>
      </div>
      <!-- End Uploading File Content -->

      <!-- Progress Bar -->
      <div class="flex w-full h-2 bg-gray-200 rounded-full overflow-hidden" role="progressbar" aria-valuenow="1" aria-valuemin="0" aria-valuemax="100">
        <div class="flex flex-col justify-center rounded-full overflow-hidden bg-blue-600 text-xs text-white text-center whitespace-nowrap transition duration-500" style="width: 1%"></div>
      </div>
      <!-- End Progress Bar -->
    </div>
  </div>
  <!-- End Body -->

  <!-- Footer -->
  <div class="bg-gray-50 border-t border-gray-200 rounded-b-xl py-2 px-4 md:px-5">
    <div class="flex flex-wrap justify-between items-center gap-x-3">
      <div>
        <span class="text-sm font-semibold text-gray-800">
          3 left
        </span>
      </div>
      <!-- End Col -->

      <div class="-me-2.5">
        <button type="button" class="py-2 px-3 inline-flex items-center gap-x-1.5 text-sm font-medium rounded-lg border border-transparent text-gray-500 hover:bg-gray-200 hover:text-gray-800 focus:outline-hidden focus:bg-gray-200 focus:text-gray-800 disabled:opacity-50 disabled:pointer-events-none">
          <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="4" height="16" x="6" y="4"></rect>
            <rect width="4" height="16" x="14" y="4"></rect>
          </svg>
          <span class="sr-only">Pause</span>
        </button>
        <button type="button" class="py-2 px-3 inline-flex items-center gap-x-1.5 text-sm font-medium rounded-lg border border-transparent text-gray-500 hover:bg-gray-200 hover:text-gray-800 focus:outline-hidden focus:bg-gray-200 focus:text-gray-800 disabled:opacity-50 disabled:pointer-events-none">
          <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 6h18"></path>
            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
            <line x1="10" x2="10" y1="11" y2="17"></line>
            <line x1="14" x2="14" y1="11" y2="17"></line>
          </svg>
          Delete
        </button>
      </div>
      <!-- End Col -->
    </div>
  </div>
  <!-- End Footer -->
</div>
<!-- End File Uploading Progress Form -->    
</template>
