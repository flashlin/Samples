<template>
  <div class="min-h-screen bg-gray-900 py-8 px-4">
    <div class="max-w-3xl mx-auto">
      <div class="bg-gray-800 rounded-lg shadow-md p-6">
        <!-- Header -->
        <div class="mb-8">
          <h1 class="text-3xl font-bold text-white mb-2">
            Build Database Model NuGet Package
          </h1>
          <p class="text-gray-400">
            Generate Entity Framework models from database and build a .NET NuGet package
          </p>
        </div>

        <!-- Form -->
        <form @submit.prevent="buildNupkg" class="space-y-6">
          <!-- Database Server -->
          <div>
            <label for="databaseServer" class="block text-sm font-medium text-gray-300 mb-2">
              Database Server
            </label>
            <input
              id="databaseServer"
              v-model="formData.databaseServer"
              type="text"
              placeholder="e.g., localhost or server.domain.com"
              class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
              required
            />
            <p class="mt-1 text-sm text-gray-400">
              Database server hostname or IP address
            </p>
          </div>

          <!-- Login ID and Password Row -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Login ID -->
            <div>
              <label for="loginId" class="block text-sm font-medium text-gray-300 mb-2">
                Login ID
              </label>
              <input
                id="loginId"
                v-model="formData.loginId"
                type="text"
                placeholder="Database username"
                class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
                required
              />
            </div>

            <!-- Login Password -->
            <div>
              <label for="loginPassword" class="block text-sm font-medium text-gray-300 mb-2">
                Login Password
              </label>
              <input
                id="loginPassword"
                v-model="formData.loginPassword"
                type="password"
                placeholder="Database password"
                class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
                required
              />
            </div>
          </div>

          <!-- Database Name -->
          <div>
            <label for="databaseName" class="block text-sm font-medium text-gray-300 mb-2">
              Database Name
            </label>
            <input
              id="databaseName"
              v-model="formData.databaseName"
              type="text"
              placeholder="e.g., MyDatabase"
              class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
              required
            />
            <p class="mt-1 text-sm text-gray-400">
              Name of the database to generate models from
            </p>
          </div>

          <!-- Namespace Name and SDK Name Row -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Namespace Name -->
            <div>
              <label for="namespaceName" class="block text-sm font-medium text-gray-300 mb-2">
                Namespace Name
              </label>
              <input
                id="namespaceName"
                v-model="formData.namespaceName"
                type="text"
                placeholder="e.g., MyApp.Models"
                class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
                required
              />
            </div>

            <!-- SDK Name -->
            <div>
              <label for="sdkName" class="block text-sm font-medium text-gray-300 mb-2">
                SDK Name
              </label>
              <input
                id="sdkName"
                v-model="formData.sdkName"
                type="text"
                placeholder="e.g., MyApp"
                class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
                required
              />
            </div>
          </div>

          <!-- SDK Version -->
          <div>
            <label for="sdkVersion" class="block text-sm font-medium text-gray-300 mb-2">
              SDK Version
            </label>
            <input
              id="sdkVersion"
              v-model="formData.sdkVersion"
              type="text"
              placeholder="1.0.0"
              class="w-full px-3 py-2 border border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white placeholder-gray-500"
            />
            <p class="mt-1 text-sm text-gray-400">
              Version number for the generated NuGet package
            </p>
          </div>

          <!-- Target Frameworks -->
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              Target Frameworks
            </label>
            <div class="space-y-2">
              <div
                v-for="framework in availableFrameworks"
                :key="framework"
                class="flex items-center"
              >
                <input
                  :id="`framework-${framework}`"
                  v-model="formData.targetFrameworks"
                  :value="framework"
                  type="checkbox"
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded bg-gray-700 border-gray-600"
                />
                <label
                  :for="`framework-${framework}`"
                  class="ml-2 block text-sm text-gray-300"
                >
                  {{ framework }}
                </label>
              </div>
            </div>
            <p class="mt-1 text-sm text-gray-400">
              Select the .NET target frameworks for the generated package
            </p>
          </div>

          <!-- Action Buttons -->
          <div class="flex items-center justify-between pt-4">
            <button
              type="button"
              @click="resetForm"
              class="px-4 py-2 border border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-300 bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Reset Form
            </button>
            
            <button
              type="submit"
              :disabled="isLoading"
              class="flex items-center px-6 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg
                v-if="isLoading"
                class="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  class="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  stroke-width="4"
                ></circle>
                <path
                  class="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              {{ isLoading ? 'Building Package...' : 'Build NuGet Package' }}
            </button>
          </div>
        </form>

        <!-- Success Message -->
        <div
          v-if="successMessage"
          class="mt-6 p-4 border border-green-400 rounded-md bg-green-900 bg-opacity-50"
        >
          <div class="flex">
            <div class="flex-shrink-0">
              <svg
                class="h-5 w-5 text-green-400"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fill-rule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clip-rule="evenodd"
                />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm font-medium text-green-400">{{ successMessage }}</p>
            </div>
          </div>
        </div>

        <!-- Error Message -->
        <div
          v-if="errorMessage"
          class="mt-6 p-4 border border-red-400 rounded-md bg-red-900 bg-opacity-50"
        >
          <div class="flex">
            <div class="flex-shrink-0">
              <svg
                class="h-5 w-5 text-red-400"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fill-rule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clip-rule="evenodd"
                />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm font-medium text-red-400">{{ errorMessage }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { codeGenApi, type BuildDatabaseModelNupkgRequest } from '../apis/codeGenApi'

// Reactive form data
const formData = reactive<BuildDatabaseModelNupkgRequest>({
  databaseServer: '',
  loginId: 'sa',
  loginPassword: '',
  databaseName: '',
  namespaceName: '',
  sdkName: '',
  sdkVersion: '1.0.0',
  targetFrameworks: ['net8.0', 'net9.0']
})

// Available framework options
const availableFrameworks = ref(['net6.0', 'net7.0', 'net8.0', 'net9.0'])

// Component state
const isLoading = ref(false)
const successMessage = ref('')
const errorMessage = ref('')

// Helper function to trigger file download
const downloadBlob = (blob: Blob, filename: string) => {
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.style.display = 'none'
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

// Build NuGet package
const buildNupkg = async () => {
  // Clear previous messages
  successMessage.value = ''
  errorMessage.value = ''

  // Validate form
  if (formData.targetFrameworks.length === 0) {
    errorMessage.value = 'Please select at least one target framework'
    return
  }

  try {
    isLoading.value = true

    // Call API to build NuGet package
    const blob = await codeGenApi.buildDatabaseModelNupkg(formData)

    // Generate filename
    const filename = `${formData.sdkName}.${formData.sdkVersion}.nupkg`

    // Trigger download
    downloadBlob(blob, filename)

    // Show success message
    successMessage.value = `Successfully built and downloaded ${filename}`
  } catch (error: any) {
    console.error('Build failed:', error)
    errorMessage.value = error.response?.data?.title || error.message || 'Failed to build NuGet package'
  } finally {
    isLoading.value = false
  }
}

// Reset form to default values
const resetForm = () => {
  formData.databaseServer = ''
  formData.loginId = 'sa'
  formData.loginPassword = ''
  formData.databaseName = ''
  formData.namespaceName = ''
  formData.sdkName = ''
  formData.sdkVersion = '1.0.0'
  formData.targetFrameworks = ['net8.0', 'net9.0']
  
  // Clear messages
  successMessage.value = ''
  errorMessage.value = ''
}
</script>
