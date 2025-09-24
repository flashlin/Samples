<script setup lang="ts">
// Setup imports for future use
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
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Welcome to CodeBoy Frontend
        </h2>
        <p class="text-gray-600 dark:text-gray-400 mb-6">
          This is a Vue3 SPA built with TypeScript, Tailwind CSS, and all the modern development tools.
        </p>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h3 class="font-semibold text-blue-900 dark:text-blue-100 mb-2">Vue3</h3>
            <p class="text-blue-700 dark:text-blue-300 text-sm">Composition API & Script Setup</p>
          </div>
          
          <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h3 class="font-semibold text-green-900 dark:text-green-100 mb-2">TypeScript</h3>
            <p class="text-green-700 dark:text-green-300 text-sm">Type safety & better DX</p>
          </div>
          
          <div class="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h3 class="font-semibold text-purple-900 dark:text-purple-100 mb-2">Tailwind CSS</h3>
            <p class="text-purple-700 dark:text-purple-300 text-sm">Utility-first CSS with dark mode</p>
          </div>
          
          <div class="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
            <h3 class="font-semibold text-orange-900 dark:text-orange-100 mb-2">Pinia</h3>
            <p class="text-orange-700 dark:text-orange-300 text-sm">State management</p>
          </div>
          
          <div class="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h3 class="font-semibold text-red-900 dark:text-red-100 mb-2">Axios</h3>
            <p class="text-red-700 dark:text-red-300 text-sm">HTTP client with interceptors</p>
          </div>
          
          <div class="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-4">
            <h3 class="font-semibold text-indigo-900 dark:text-indigo-100 mb-2">Mock.js</h3>
            <p class="text-indigo-700 dark:text-indigo-300 text-sm">API mocking for development</p>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>
