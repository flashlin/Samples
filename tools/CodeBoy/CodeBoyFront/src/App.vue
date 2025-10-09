<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import DropdownSearch, { type SearchItem } from './components/DropdownSearch.vue'

const route = useRoute()
const router = useRouter()

interface NavItem extends SearchItem {
  path: string
}

const navItems: NavItem[] = [
  { path: '/', name: 'Code Generator', description: 'Generate Web API client code' },
  { path: '/buildSwaggerCSharpSdk', name: 'Build C# SDK', description: 'Build Swagger C# SDK NuGet package' },
  { path: '/buildDatabaseModelNupkg', name: 'Build DB Models', description: 'Build database model NuGet package' },
  { path: '/useGrpcProtoGeneratorDoc', name: 'gRPC Proto Docs', description: 'gRPC Proto generator documentation' },
  { path: '/genTypescriptCodeFromSwagger', name: 'Gen TypeScript', description: 'Generate TypeScript API client' },
  { path: '/genDatabaseDto', name: 'Gen Database DTO', description: 'Generate database DTO from SQL' },
  { path: '/genCodeFirstFromDatabase', name: 'Gen EF Code First', description: 'Generate EF Code First models from database' },
  { path: '/genProtoCodeFromGrpcClientAssembly', name: 'Gen Proto from gRPC Client SDK', description: 'Generate proto code from gRPC assembly' }
]

const selectedItem = ref<NavItem | undefined>(
  navItems.find(item => item.path === route.path)
)

const handleItemSelect = (item: SearchItem) => {
  const navItem = item as NavItem
  selectedItem.value = navItem
  router.push(navItem.path)
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
            
            <DropdownSearch 
              :items="navItems" 
              :selected-value="selectedItem"
              placeholder="搜尋功能..." 
              @select="handleItemSelect" 
            />
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
