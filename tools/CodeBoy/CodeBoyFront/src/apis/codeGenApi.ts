import request from './request';

// Request/Response interfaces for code generation
export interface GenWebApiClientRequest {
  swaggerUrl: string;
  sdkName: string;
}

export interface GenWebApiClientResponse {
  generatedCode: string;
  success: boolean;
  errorMessage?: string;
}

// Code generation API
export const codeGenApi = {
  /**
   * Generate Web API client code from Swagger URL
   * @param params Generation request parameters
   * @returns Generated client code
   */
  generateWebApiClient(params: GenWebApiClientRequest): Promise<GenWebApiClientResponse> {
    return request.post('/codegen/genWebApiClient', params);
  },
};