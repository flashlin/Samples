<template>
  <div class="min-h-screen bg-gray-900 py-8 px-4">
    <div class="max-w-2xl mx-auto">
      <div class="bg-gray-800 rounded-lg shadow-md p-6">
        <!-- Header -->
        <div class="mb-8">
          <h1 class="text-3xl font-bold text-white mb-2">
            Generate TypeScript Code from Swagger
          </h1>
          <p class="text-gray-400">
            Generate TypeScript API client code from Swagger/OpenAPI documentation
          </p>
        </div>

        <!-- Form -->
        <form @submit.prevent="generateCode" class="space-y-6">
          <!-- API Name -->
          <div>
            <label for="apiName" class="block text-sm font-medium text-gray-300 mb-2">
              API Name
            </label>
            <input
              id="apiName"
              v-model="formData.apiName"
              type="text"
              placeholder="e.g., myApi"
              class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
              required
            />
            <p class="mt-1 text-sm text-gray-400">
              The name of the generated API client object
            </p>
          </div>

          <!-- Swagger URL -->
          <div>
            <label for="swaggerUrl" class="block text-sm font-medium text-gray-300 mb-2">
              Swagger URL
            </label>
            <input
              id="swaggerUrl"
              v-model="formData.swaggerUrl"
              type="url"
              placeholder="https://api.example.com/swagger/v1/swagger.json"
              class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
              required
            />
            <p class="mt-1 text-sm text-gray-400">
              URL to the Swagger/OpenAPI JSON specification
            </p>
          </div>

          <!-- Generate Button -->
          <div class="pt-4">
            <button
              type="submit"
              :disabled="isGenerating"
              class="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <span v-if="isGenerating" class="flex items-center justify-center">
                <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating...
              </span>
              <span v-else>Generate TypeScript Code</span>
            </button>
          </div>
        </form>

        <!-- Error Message -->
        <div v-if="errorMessage" class="mt-6 p-4 bg-red-900/20 border border-red-800 rounded-md">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-red-200">Generation Failed</h3>
              <p class="mt-1 text-sm text-red-300">{{ errorMessage }}</p>
            </div>
          </div>
        </div>

        <!-- Generated Code Display -->
        <div v-if="generatedCode" class="mt-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-medium text-white">Generated TypeScript Code</h3>
            <button
              @click="copyToClipboard"
              class="px-3 py-2 bg-gray-700 text-gray-300 rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
            >
              <span v-if="copied" class="flex items-center">
                <svg class="w-4 h-4 mr-2 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                Copied!
              </span>
              <span v-else class="flex items-center">
                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                </svg>
                Copy Code
              </span>
            </button>
          </div>
          <div class="bg-gray-900 rounded-lg border border-gray-600 overflow-hidden">
            <pre class="p-4 text-sm text-green-400 overflow-x-auto"><code>{{ generatedCode }}</code></pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { codeGenApi, type GenTypescriptCodeFromSwaggerRequest } from '@/apis/codeGenApi'
import { LocalStorageService } from '@/services/LocalStorage'

// Form data
const formData = reactive<GenTypescriptCodeFromSwaggerRequest>({
  apiName: '',
  swaggerUrl: ''
})

// State
const isGenerating = ref(false)
const errorMessage = ref('')
const generatedCode = ref('')
const copied = ref(false)

// Clear messages
const clearMessages = () => {
  errorMessage.value = ''
  generatedCode.value = ''
}

// Copy to clipboard
const copyToClipboard = async () => {
  try {
    await navigator.clipboard.writeText(generatedCode.value)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  } catch (error) {
    console.error('Failed to copy to clipboard:', error)
  }
}

// Generate TypeScript code
const generateCode = async () => {
  clearMessages()
  isGenerating.value = true

  try {
    // Call API to generate TypeScript code
    const code = await codeGenApi.genTypescriptCodeFromSwagger(formData)
    generatedCode.value = code
  } catch (error: any) {
    console.error('Generation failed:', error)
    errorMessage.value = error.response?.data?.detail || error.message || 'Failed to generate TypeScript code. Please check your inputs and try again.'
  } finally {
    isGenerating.value = false
  }
}

// Storage key for form data
const STORAGE_KEY = 'genTypescriptCodeFromSwagger_formData'

// Load form data from storage on mount
onMounted(async () => {
  try {
    const savedData = await LocalStorageService.loadFromStorage<GenTypescriptCodeFromSwaggerRequest>(STORAGE_KEY)
    if (savedData) {
      formData.apiName = savedData.apiName || ''
      formData.swaggerUrl = savedData.swaggerUrl || ''
    }
  } catch (error) {
    console.error('Failed to load form data from storage:', error)
  }
})

// Save form data to storage on unmount
onUnmounted(async () => {
  try {
    await LocalStorageService.saveToStorage(STORAGE_KEY, formData)
  } catch (error) {
    console.error('Failed to save form data to storage:', error)
  }
})
</script>

<style scoped>
/* Custom scrollbar for code blocks */
pre::-webkit-scrollbar {
  height: 8px;
}

pre::-webkit-scrollbar-track {
  background: #1f2937;
}

pre::-webkit-scrollbar-thumb {
  background: #4b5563;
  border-radius: 4px;
}

pre::-webkit-scrollbar-thumb:hover {
  background: #6b7280;
}

/* Ensure proper font for code */
pre code {
  font-family: 'Courier New', Courier, monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>
