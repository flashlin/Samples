import request from './request';

// Request interface for code generation
export interface GenWebApiClientRequest {
  swaggerUrl: string;
  sdkName: string;
}

// Request interface for building nupkg
export interface BuildWebApiClientNupkgRequest {
  sdkName: string;
  swaggerUrl: string;
  nupkgName: string;
  sdkVersion: string;
}

// Request interface for building database model nupkg
export interface BuildDatabaseModelNupkgRequest {
  databaseServer: string;
  loginId: string;
  loginPassword: string;
  databaseName: string;
  namespaceName: string;
  sdkName: string;
  sdkVersion: string;
  targetFrameworks: string[];
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

  /**
   * Build Web API client nupkg from Swagger URL
   * @param params Build request parameters
   * @returns File download blob
   */
  buildWebApiClientNupkg(params: BuildWebApiClientNupkgRequest): Promise<Blob> {
    return request.post('/codegen/buildWebApiClientNupkg', params, {
      responseType: 'blob'
    });
  },

  /**
   * Build database model nupkg from database connection
   * @param params Build request parameters
   * @returns File download blob
   */
  buildDatabaseModelNupkg(params: BuildDatabaseModelNupkgRequest): Promise<Blob> {
    return request.post('/codegen/buildDatabaseModelNupkg', params, {
      responseType: 'blob'
    });
  },
};