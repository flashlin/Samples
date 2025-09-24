<script setup lang="ts">
import CodeGenerator from '@/views/CodeGenerator.vue'
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    <header class="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <h1 class="text-xl font-semibold text-gray-900 dark:text-white">
            CodeBoy Frontend
          </h1>
          <div class="flex items-center space-x-4">
            <span class="text-sm text-gray-600 dark:text-gray-400">
              Vue3 + TypeScript + Tailwind
            </span>
          </div>
        </div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
      <CodeGenerator />
    </main>
  </div>
</template>
