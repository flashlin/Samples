<template>
  <div class="min-h-screen p-12 bg-gray-900 text-white">
    <div class="max-w-4xl mx-auto space-y-12">
      <h1 class="text-4xl font-bold text-blue-400 border-b border-gray-800 pb-4">
        {{ store.title }}
      </h1>

      <section class="grid grid-cols-1 md:grid-cols-2 gap-8">
        <!-- DropDownList Demo -->
        <div class="p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
          <h2 class="text-xl font-semibold text-blue-300 mb-4">DropDownList</h2>
          <p class="text-sm text-gray-400 mb-6">Enforced selection from list.</p>
          
          <DropDownList
            v-model="selectedTech"
            :options="techOptions"
            placeholder="Select a technology..."
          />
          
          <div class="mt-4 p-2 bg-gray-900 rounded border border-gray-700 text-xs font-mono">
           Value: <span class="text-blue-400">{{ selectedTech || 'unset' }}</span>
          </div>
        </div>

        <!-- AutoComplete Demo -->
        <div class="p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
          <h2 class="text-xl font-semibold text-emerald-300 mb-4">AutoComplete</h2>
          <p class="text-sm text-gray-400 mb-6">Free input with suggestions.</p>
          
          <AutoComplete
            v-model="searchQuery"
            :options="searchOptions"
            placeholder="Search or type freely..."
          />
          
          <div class="mt-4 p-2 bg-gray-900 rounded border border-gray-700 text-xs font-mono">
            Value: <span class="text-emerald-400">{{ searchQuery || '""' }}</span>
          </div>
        </div>
      </section>

      <!-- JsonEditor Demo -->
      <section class="p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
        <h2 class="text-xl font-semibold text-purple-300 mb-4">JsonEditor</h2>
        <p class="text-sm text-gray-400 mb-6">Dynamic array editor with enforced schema and CRUD operations.</p>
        
        <JsonEditor
          v-model="userList"
          :schema="userSchema"
          @change="onJsonChange"
        />
        
        <div class="mt-6 p-4 bg-gray-900 rounded border border-gray-700">
          <h3 class="text-xs font-medium text-gray-500 uppercase mb-2">Current JSON Data</h3>
          <pre class="text-xs font-mono text-purple-400 overflow-x-auto">{{ JSON.stringify(userList, null, 2) }}</pre>
        </div>
      </section>

      <div class="p-6 bg-gray-800/50 rounded-lg italic text-gray-400 text-center">
        Try "p" to find Pnpm, ViteBuildTool, or "v" to find Vue.js etc (Camel Search enabled).
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useAppStore } from '../store'

const store = useAppStore()

const selectedTech = ref('')
const techOptions = [
  { label: 'Vue.js', value: 'vue' },
  { label: 'TypeScript', value: 'ts' },
  { label: 'TailwindCSS', value: 'tailwind' },
  { label: 'ViteBuildTool', value: 'vite' },
  { label: 'PnpmManager', value: 'pnpm' }
]

const searchQuery = ref('')
const searchOptions = [
  'ViteBuildTool',
  'VueComponents',
  'TailwindLayouts',
  'TypeScriptIntegration',
  'PiniaState'
]

// JsonEditor Demo
const userList = ref([
  { id: 1, name: 'John Doe', birth: '1990-01-01' },
  { id: 2, name: 'Jane Smith', birth: '1995-05-15' }
])

const userSchema = [
  { key: 'id', label: 'ID', type: 'number' },
  { key: 'name', label: 'Full Name', type: 'string' },
  { key: 'birth', label: 'Birth Date', type: 'date' }
]

const onJsonChange = (data: any) => {
  console.log('JSON Data Changed:', data)
}
</script>
