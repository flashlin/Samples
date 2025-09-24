import { defineStore } from 'pinia';
import { ref } from 'vue';
import { codeGenApi, type GenWebApiClientRequest, type GenWebApiClientResponse } from '@/apis/codeGenApi';

export const useCodeGenStore = defineStore('codeGen', () => {
  // State
  const loading = ref(false);
  const generatedCode = ref<string>('');
  const lastRequest = ref<GenWebApiClientRequest | null>(null);
  const error = ref<string | null>(null);

  // Actions
  const generateWebApiClient = async (params: GenWebApiClientRequest): Promise<GenWebApiClientResponse> => {
    loading.value = true;
    error.value = null;
    
    try {
      const result = await codeGenApi.generateWebApiClient(params);
      
      if (result.success) {
        generatedCode.value = result.generatedCode;
        lastRequest.value = params;
      } else {
        error.value = result.errorMessage || 'Code generation failed';
      }
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      error.value = errorMessage;
      console.error('Code generation failed:', err);
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const clearGeneratedCode = () => {
    generatedCode.value = '';
    lastRequest.value = null;
    error.value = null;
  };

  const clearError = () => {
    error.value = null;
  };

  return {
    // State
    loading,
    generatedCode,
    lastRequest,
    error,
    // Actions
    generateWebApiClient,
    clearGeneratedCode,
    clearError,
  };
});
