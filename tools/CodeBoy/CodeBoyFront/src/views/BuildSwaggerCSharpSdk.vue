<template>
  <div class="min-h-screen bg-gray-50 py-8 px-4">
    <div class="max-w-2xl mx-auto">
      <div class="bg-white rounded-lg shadow-md p-6">
        <!-- Header -->
        <div class="mb-8">
          <h1 class="text-3xl font-bold text-gray-900 mb-2">
            Build Swagger C# SDK
          </h1>
          <p class="text-gray-600">
            Generate and build a .NET SDK NuGet package from Swagger/OpenAPI documentation
          </p>
        </div>

        <!-- Form -->
        <form @submit.prevent="buildNupkg" class="space-y-6">
          <!-- SDK Name -->
          <div>
            <label for="sdkName" class="block text-sm font-medium text-gray-700 mb-2">
              SDK Name
            </label>
            <input
              id="sdkName"
              v-model="formData.sdkName"
              type="text"
              placeholder="e.g., MyApiClient"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              required
            />
            <p class="mt-1 text-sm text-gray-500">
              The name of the generated SDK client class
            </p>
          </div>

          <!-- Swagger URL -->
          <div>
            <label for="swaggerUrl" class="block text-sm font-medium text-gray-700 mb-2">
              Swagger URL
            </label>
            <input
              id="swaggerUrl"
              v-model="formData.swaggerUrl"
              type="url"
              placeholder="https://api.example.com/swagger/v1/swagger.json"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              required
            />
            <p class="mt-1 text-sm text-gray-500">
              URL to the Swagger/OpenAPI JSON specification
            </p>
          </div>

          <!-- NuPkg Name -->
          <div>
            <label for="nupkgName" class="block text-sm font-medium text-gray-700 mb-2">
              NuGet Package Name
            </label>
            <input
              id="nupkgName"
              v-model="formData.nupkgName"
              type="text"
              placeholder="e.g., MyCompany.MyApiClient"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              required
            />
            <p class="mt-1 text-sm text-gray-500">
              The name for the generated NuGet package
            </p>
          </div>

          <!-- Build Button -->
          <div class="pt-4">
            <button
              type="submit"
              :disabled="isBuilding"
              class="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <span v-if="isBuilding" class="flex items-center justify-center">
                <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Building SDK...
              </span>
              <span v-else>Build SDK Package</span>
            </button>
          </div>
        </form>

        <!-- Error Message -->
        <div v-if="errorMessage" class="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-red-800">Build Failed</h3>
              <p class="mt-1 text-sm text-red-700">{{ errorMessage }}</p>
            </div>
          </div>
        </div>

        <!-- Success Message -->
        <div v-if="successMessage" class="mt-6 p-4 bg-green-50 border border-green-200 rounded-md">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-green-800">Success!</h3>
              <p class="mt-1 text-sm text-green-700">{{ successMessage }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { codeGenApi, type BuildWebApiClientNupkgRequest } from '@/apis/codeGenApi'

// Form data
const formData = reactive<BuildWebApiClientNupkgRequest>({
  sdkName: '',
  swaggerUrl: '',
  nupkgName: ''
})

// State
const isBuilding = ref(false)
const errorMessage = ref('')
const successMessage = ref('')

// Clear messages
const clearMessages = () => {
  errorMessage.value = ''
  successMessage.value = ''
}

// Download file helper
const downloadFile = (blob: Blob, filename: string) => {
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}

// Build NuPkg
const buildNupkg = async () => {
  clearMessages()
  isBuilding.value = true

  try {
    // Call API to build the NuPkg
    const blob = await codeGenApi.buildWebApiClientNupkg(formData)
    
    // Generate filename for download
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '')
    const filename = `${formData.nupkgName}_${timestamp}.nupkg`
    
    // Download the file
    downloadFile(blob, filename)
    
    successMessage.value = `SDK package "${filename}" has been generated and downloaded successfully!`
  } catch (error: any) {
    console.error('Build failed:', error)
    errorMessage.value = error.response?.data?.detail || error.message || 'Failed to build SDK package. Please check your inputs and try again.'
  } finally {
    isBuilding.value = false
  }
}
</script>

<style scoped>
/* Additional custom styles if needed */
</style>
