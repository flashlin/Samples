<template>
  <div class="min-h-screen bg-gray-900 text-white p-12 font-sans">
    <div class="max-w-4xl mx-auto space-y-12">
      <header class="border-b border-gray-800 pb-6">
        <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
          T1 Web Components
        </h1>
        <p class="mt-2 text-gray-400">基於 Vue 3 + Tailwind 3 的高品質暗黑模式元件庫</p>
      </header>

      <section class="grid grid-cols-1 md:grid-cols-2 gap-12">
        <!-- DropDownList 展示 -->
        <div class="space-y-4 p-6 bg-gray-800/50 rounded-xl border border-gray-700 shadow-lg">
          <h2 class="text-xl font-bold flex items-center gap-2">
            <span class="w-2 h-6 bg-blue-500 rounded-full"></span>
            DropDownList
          </h2>
          <p class="text-sm text-gray-400 mb-4">強制選取清單內容，支援 Camel Case 搜尋與鍵盤導航。</p>
          
          <DropDownList
            v-model="dropDownValue"
            :options="dropDownOptions"
            placeholder="請選擇一個技術..."
          />
          
          <div class="mt-6 p-3 bg-gray-900 rounded-md border border-gray-700 text-xs font-mono">
            <span class="text-gray-500">Selected Value:</span>
            <span class="text-blue-400 ml-2">{{ dropDownValue || 'null' }}</span>
          </div>
        </div>

        <!-- AutoComplete 展示 -->
        <div class="space-y-4 p-6 bg-gray-800/50 rounded-xl border border-gray-700 shadow-lg">
          <h2 class="text-xl font-bold flex items-center gap-2">
            <span class="w-2 h-6 bg-emerald-500 rounded-full"></span>
            AutoComplete
          </h2>
          <p class="text-sm text-gray-400 mb-4">允許自由輸入，智慧推薦清單內容並支援高亮顯示。</p>
          
          <AutoComplete
            v-model="autoCompleteValue"
            :options="autoCompleteOptions"
            placeholder="搜尋關鍵字或隨意輸入..."
          />
          
          <div class="mt-6 p-3 bg-gray-900 rounded-md border border-gray-700 text-xs font-mono">
            <span class="text-gray-500">Selected/Input Value:</span>
            <span class="text-emerald-400 ml-2">{{ autoCompleteValue || '""' }}</span>
          </div>
        </div>
      </section>

      <!-- JsonEditor 展示 (Array) -->
      <section class="p-6 bg-gray-800/50 rounded-xl border border-gray-700 shadow-lg">
        <h2 class="text-xl font-bold flex items-center gap-2 mb-2">
          <span class="w-2 h-6 bg-purple-500 rounded-full"></span>
          JsonEditor (Array Mode)
        </h2>
        <p class="text-sm text-gray-400 mb-6">動態陣列編輯器，支援 schema 驗證與完整的 CRUD 操作。使用 JSON 字串格式，支援格式化輸出。</p>

        <JsonEditor
          v-model="userListJson"
          :schema="userSchema"
          :compact="false"
          @error="handleJsonError"
        />

        <div class="mt-6 p-4 bg-gray-900 rounded-md border border-gray-700">
          <div class="flex justify-between items-center mb-2">
            <h3 class="text-xs font-medium text-gray-500 uppercase">Current JSON String</h3>
            <span class="text-xs text-gray-500">{{ userListJson.length }} chars</span>
          </div>
          <pre class="text-xs font-mono text-purple-400 overflow-x-auto whitespace-pre-wrap">{{ userListJson }}</pre>
        </div>
      </section>

      <!-- JsonEditor 展示 (Object) -->
      <section class="p-6 bg-gray-800/50 rounded-xl border border-gray-700 shadow-lg border-l-4 border-l-orange-500">
        <h2 class="text-xl font-bold flex items-center gap-2 mb-2">
          <span class="w-2 h-6 bg-orange-500 rounded-full"></span>
          JsonEditor (Object Mode)
        </h2>
        <p class="text-sm text-gray-400 mb-6">
          單一物件編輯器，自動切換為表單佈局。需按 Save 才會更新 modelValue，
          <span class="text-orange-300 font-semibold">schema 自動從 JSON 推斷（extractSchema）</span>。
        </p>

        <JsonEditor
          v-model="singleUserJson"
          :compact="true"
          @error="handleJsonError"
        />

        <div class="mt-6 p-4 bg-gray-900 rounded-md border border-gray-700">
          <div class="flex justify-between items-center mb-2">
            <h3 class="text-xs font-medium text-gray-500 uppercase">Current JSON String (Compact)</h3>
            <span class="text-xs text-gray-500">{{ singleUserJson.length }} chars</span>
          </div>
          <pre class="text-xs font-mono text-orange-400 overflow-x-auto whitespace-pre-wrap">{{ singleUserJson }}</pre>
        </div>
      </section>

      <section class="p-6 bg-blue-900/20 rounded-xl border border-blue-800/50">
        <h3 class="text-lg font-semibold text-blue-300 mb-2">搜尋提示 (Camel Search)</h3>
        <ul class="text-sm text-gray-300 list-disc list-inside space-y-1">
          <li>試著輸入 <code class="bg-gray-800 px-1 rounded text-yellow-400">VC</code> 可匹配 <code class="text-white font-bold">V</code>ue<code class="text-white font-bold">C</code>omponents</li>
          <li>輸入 <code class="bg-gray-800 px-1 rounded text-yellow-400">TW</code> 可匹配 <code class="text-white font-bold">T</code>ail<code class="text-white font-bold">w</code>ind</li>
          <li>使用方向鍵 <code class="text-gray-400 font-bold">↑ ↓</code> 選取，按 <code class="text-gray-400 font-bold">ENTER</code> 確認</li>
        </ul>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import DropDownList from './components/DropDownList.vue'
import AutoComplete from './components/AutoComplete.vue'
import JsonEditor from './components/JsonEditor.vue'

// DropDownList 資料
const dropDownValue = ref('')
const dropDownOptions = [
  { label: 'Vue.js', value: 'vue' },
  { label: 'TypeScript', value: 'ts' },
  { label: 'TailwindCSS', value: 'tailwind' },
  { label: 'ViteBuildTool', value: 'vite' },
  { label: 'PnpmManager', value: 'pnpm' },
  { label: 'ProjectSetup', value: 'setup' },
  { label: 'WebComponents', value: 'web-components' }
]

// AutoComplete 資料
const autoCompleteValue = ref('')
const autoCompleteOptions = [
  'ReactFramework',
  'AngularFramework',
  'SvelteFramework',
  'NextJsApp',
  'NuxtJsApp',
  'TailwindAesthetics',
  'ModernDesign'
]

// JsonEditor 資料 (JSON 字串格式)
const userListJson = ref<string>(`[
  {"id": 1, "name": "John Doe", "birth": "1990-01-01"},
  {"id": 2, "name": "Jane Smith", "birth": "1995-05-15"},
  {"id": 3, "name": "Bob Johnson", "birth": "1988-03-20"}
]`)

const userSchema = [
  { key: 'id', label: 'ID', type: 'number' as const },
  { key: 'name', label: 'Full Name', type: 'string' as const },
  { key: 'birth', label: 'Birth Date', type: 'date' as const }
]

// JsonEditor 單一物件資料 (JSON 字串格式)
const singleUserJson = ref<string>(`{"cid":123,"name":"flash","birth":"2005-11-01T12:00:00-04:00"}`)

// Error handler
const handleJsonError = (message: string) => {
  console.error('JSON Editor Error:', message)
}
</script>

<style>
/* 隱藏捲軸樣式 - 範例使用 */
.scrollbar-thin::-webkit-scrollbar {
  width: 6px;
}
.scrollbar-thin::-webkit-scrollbar-track {
  @apply bg-transparent;
}
.scrollbar-thin::-webkit-scrollbar-thumb {
  @apply bg-gray-600 rounded-full;
}
</style>
