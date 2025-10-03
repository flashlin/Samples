<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()

const searchQuery = ref('')
const isSearchFocused = ref(false)

interface NavItem {
  path: string
  name: string
  description: string
}

const navItems: NavItem[] = [
  { path: '/', name: 'Code Generator', description: 'Generate Web API client code' },
  { path: '/buildSwaggerCSharpSdk', name: 'Build C# SDK', description: 'Build Swagger C# SDK NuGet package' },
  { path: '/buildDatabaseModelNupkg', name: 'Build DB Models', description: 'Build database model NuGet package' },
  { path: '/useGrpcProtoGeneratorDoc', name: 'gRPC Proto Docs', description: 'gRPC Proto generator documentation' },
  { path: '/genTypescriptCodeFromSwagger', name: 'Gen TypeScript', description: 'Generate TypeScript API client' },
  { path: '/genDatabaseDto', name: 'Gen Database DTO', description: 'Generate database DTO from SQL' },
  { path: '/genProtoCodeFromGrpcClientAssembly', name: 'Gen Proto from gRPC Client SDK', description: 'Generate proto code from gRPC assembly' }
]

const filteredNavItems = computed(() => {
  if (!searchQuery.value.trim()) {
    return navItems
  }
  const query = searchQuery.value.toLowerCase()
  return navItems.filter(item => 
    item.name.toLowerCase().includes(query) || 
    item.description.toLowerCase().includes(query)
  )
})

const showNavList = computed(() => {
  return isSearchFocused.value
})

const handleSearchFocus = () => {
  isSearchFocused.value = true
}

const handleSearchBlur = () => {
  setTimeout(() => {
    isSearchFocused.value = false
  }, 200)
}

const navigateTo = (path: string) => {
  router.push(path)
  searchQuery.value = ''
  isSearchFocused.value = false
}
</script>

<template>
  <div class="min-h-screen bg-gray-900 text-gray-100">
    <header class="bg-gray-800 shadow-sm border-b border-gray-700">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <div class="flex items-center space-x-8 flex-1">
            <h1 class="text-xl font-semibold text-white whitespace-nowrap">
              CodeBoy
            </h1>
            
            <!-- Search Navigation -->
            <div class="relative flex-1 max-w-md">
              <div class="relative">
                <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                  </svg>
                </div>
                <input
                  v-model="searchQuery"
                  type="text"
                  placeholder="搜尋功能..."
                  class="w-full pl-10 pr-3 py-2 border border-gray-600 rounded-md bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  @focus="handleSearchFocus"
                  @blur="handleSearchBlur"
                />
              </div>

              <!-- Navigation List -->
              <div
                v-if="showNavList"
                class="absolute z-50 mt-2 w-full bg-gray-800 border border-gray-600 rounded-md shadow-lg max-h-96 overflow-y-auto"
              >
                <div v-if="filteredNavItems.length === 0" class="px-4 py-3 text-sm text-gray-400">
                  找不到符合的功能
                </div>
                <div v-else>
                  <button
                    v-for="item in filteredNavItems"
                    :key="item.path"
                    @click="navigateTo(item.path)"
                    class="w-full text-left px-4 py-3 hover:bg-gray-700 transition-colors border-b border-gray-700 last:border-b-0"
                    :class="route.path === item.path ? 'bg-blue-900/30' : ''"
                  >
                    <div class="flex items-center justify-between">
                      <div>
                        <div class="text-sm font-medium text-white">{{ item.name }}</div>
                        <div class="text-xs text-gray-400 mt-1">{{ item.description }}</div>
                      </div>
                      <svg
                        v-if="route.path === item.path"
                        class="h-5 w-5 text-blue-400 flex-shrink-0 ml-2"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                      </svg>
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          <div class="flex items-center space-x-4 ml-4">
            <span class="text-sm text-gray-400 whitespace-nowrap">
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
