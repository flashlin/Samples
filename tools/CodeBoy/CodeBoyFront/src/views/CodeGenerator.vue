<template>
  <div class="max-w-4xl mx-auto py-8 px-4">
    <div class="bg-gray-800 rounded-lg shadow-sm border border-gray-700">
      <!-- Header -->
      <div class="border-b border-gray-700 px-6 py-4">
        <h1 class="text-2xl font-bold text-white">
          Web API Client Generator
        </h1>
        <p class="text-gray-400 mt-1">
          Generate C# client code from Swagger/OpenAPI specifications
        </p>
      </div>

      <!-- Form -->
      <div class="p-6">
        <form @submit.prevent="handleGenerate" class="space-y-6">
          <!-- Swagger URL Input -->
          <div>
            <label for="swaggerUrl" class="block text-sm font-medium text-gray-300 mb-2">
              Swagger URL
            </label>
            <input
              id="swaggerUrl"
              v-model="form.swaggerUrl"
              type="url"
              required
              placeholder="https://api.example.com/swagger/v1/swagger.json"
              class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm
                     focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white
                     placeholder-gray-500"
            />
          </div>

          <!-- SDK Name Input -->
          <div>
            <label for="sdkName" class="block text-sm font-medium text-gray-300 mb-2">
              SDK Name
            </label>
            <input
              id="sdkName"
              v-model="form.sdkName"
              type="text"
              required
              placeholder="MyApiClient"
              class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm
                     focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white
                     placeholder-gray-500"
            />
          </div>

          <!-- Generate Button -->
          <div>
            <button
              type="submit"
              :disabled="loading"
              class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium
                     py-2 px-4 rounded-md transition-colors duration-200 flex items-center justify-center"
            >
              <svg v-if="loading" class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {{ loading ? 'Generating...' : 'Generate Code' }}
            </button>
          </div>
        </form>

        <!-- Error Display -->
        <div v-if="error" class="mt-6 p-4 bg-red-900/20 border border-red-800 rounded-md">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-red-200">
                Generation Failed
              </h3>
              <div class="mt-2 text-sm text-red-300">
                {{ error }}
              </div>
            </div>
            <div class="ml-auto pl-3">
              <button @click="clearError" class="text-red-400 hover:text-red-600">
                <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Generated Code Display -->
    <div v-if="generatedCode" class="mt-8 bg-gray-800 rounded-lg shadow-sm border border-gray-700">
      <div class="border-b border-gray-700 px-6 py-4 flex justify-between items-center">
        <h2 class="text-lg font-semibold text-white">
          Generated Code
        </h2>
        <div class="flex space-x-2">
          <button
            @click="copyToClipboard"
            class="bg-gray-700 hover:bg-gray-600 text-gray-300 px-3 py-1 rounded text-sm transition-colors"
          >
            Copy
          </button>
          <button
            @click="downloadCode"
            class="bg-blue-900/30 hover:bg-blue-900/50 text-blue-300 px-3 py-1 rounded text-sm transition-colors"
          >
            Download
          </button>
        </div>
      </div>
      <div class="p-6">
        <pre class="bg-gray-900 p-4 rounded-md overflow-x-auto text-sm"><code class="language-csharp text-gray-100">{{ generatedCode }}</code></pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { codeGenApi } from '@/apis/codeGenApi'
import { LocalStorageService } from '@/services/LocalStorage'

// Form data
const form = ref({
  swaggerUrl: '',
  sdkName: ''
})

// State
const loading = ref(false)
const generatedCode = ref('')
const error = ref<string | null>(null)

// Methods
const handleGenerate = async () => {
  loading.value = true
  error.value = null
  
  try {
    const result = await codeGenApi.generateWebApiClient({
      swaggerUrl: form.value.swaggerUrl,
      sdkName: form.value.sdkName
    })
    
    generatedCode.value = result
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'
    error.value = errorMessage
    console.error('Generation failed:', err)
  } finally {
    loading.value = false
  }
}

const clearError = () => {
  error.value = null
}

const copyToClipboard = async () => {
  try {
    await navigator.clipboard.writeText(generatedCode.value)
    // TODO: Show success toast
  } catch (err) {
    console.error('Failed to copy:', err)
  }
}

const downloadCode = () => {
  const blob = new Blob([generatedCode.value], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${form.value.sdkName || 'GeneratedClient'}.cs`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// Storage key for form data
const STORAGE_KEY = 'codeGenerator_formData'

// Load form data from storage on mount
onMounted(async () => {
  try {
    const savedData = await LocalStorageService.loadFromStorage<typeof form.value>(STORAGE_KEY)
    if (savedData) {
      form.value = savedData
    }
  } catch (error) {
    console.error('Failed to load form data from storage:', error)
  }
})

// Save form data to storage on unmount
onUnmounted(async () => {
  try {
    await LocalStorageService.saveToStorage(STORAGE_KEY, form.value)
  } catch (error) {
    console.error('Failed to save form data to storage:', error)
  }
})
</script>
