<script setup lang="ts">
import { useRoute } from 'vue-router'

const route = useRoute()
</script>

<template>
  <div class="min-h-screen bg-gray-900 text-gray-100">
    <header class="bg-gray-800 shadow-sm border-b border-gray-700">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <div class="flex items-center space-x-8">
            <h1 class="text-xl font-semibold text-white">
              CodeBoy
            </h1>
            
            <!-- Navigation -->
            <nav class="hidden md:flex space-x-6">
              <RouterLink
                to="/"
                class="px-3 py-2 text-sm font-medium rounded-md transition-colors"
                :class="route.path === '/' 
                  ? 'bg-blue-900 text-blue-300' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'"
              >
                Code Generator
              </RouterLink>
              <RouterLink
                to="/buildSwaggerCSharpSdk"
                class="px-3 py-2 text-sm font-medium rounded-md transition-colors"
                :class="route.path === '/buildSwaggerCSharpSdk' 
                  ? 'bg-blue-900 text-blue-300' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'"
              >
                Build C# SDK
              </RouterLink>
              <RouterLink
                to="/buildDatabaseModelNupkg"
                class="px-3 py-2 text-sm font-medium rounded-md transition-colors"
                :class="route.path === '/buildDatabaseModelNupkg' 
                  ? 'bg-blue-900 text-blue-300' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'"
              >
                Build DB Models
              </RouterLink>
            </nav>
          </div>
          
          <div class="flex items-center space-x-4">
            <span class="text-sm text-gray-400">
              v1.0
            </span>
          </div>
        </div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
      <RouterView />
    </main>
  </div>
</template>
