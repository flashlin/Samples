import request from './request';

// Request interface for code generation
export interface GenWebApiClientRequest {
  swaggerUrl: string;
  sdkName: string;
}

// Code generation API
export const codeGenApi = {
  /**
   * Generate Web API client code from Swagger URL
   * @param params Generation request parameters
   * @returns Generated client code as string
   */
  generateWebApiClient(params: GenWebApiClientRequest): Promise<string> {
    return request.post('/codegen/genWebApiClient', params);
  },
};