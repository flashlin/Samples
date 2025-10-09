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

// Request interface for generating TypeScript code from Swagger
export interface GenTypescriptCodeFromSwaggerRequest {
  apiName: string;
  swaggerUrl: string;
}

// Request interface for generating database DTO code
export interface GenDatabaseDtoRequest {
  sql: string;
}

// Request interface for generating EF Code First from database
export interface GenCodeFirstFromDatabaseRequest {
  databaseServer: string;
  loginId: string;
  loginPassword: string;
  databaseName: string;
  namespaceName: string;
  targetFramework: string;
}

// Response interface for EF Code First generation
export interface EfGeneratedFile {
  fileName: string;
  fileContent: string;
}

export interface EfGenerationOutput {
  csprojFilePath: string;
  codeFiles: EfGeneratedFile[];
}

// Request interface for generating proto code from gRPC client assembly
export interface GenProtoCodeFromGrpcClientAssemblyRequest {
  namespaceName: string;
  assembly: Uint8Array;
}

// Response interface for proto code generation
export interface ProtoFileInfo {
  serviceName: string;
  protoCode: string;
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

  /**
   * Generate TypeScript API client code from Swagger URL
   * @param params TypeScript generation request parameters
   * @returns Generated TypeScript code as string
   */
  genTypescriptCodeFromSwagger(params: GenTypescriptCodeFromSwaggerRequest): Promise<string> {
    return request.post('/codegen/genTypescriptCodeFromSwagger', params);
  },

  /**
   * Generate database DTO code from SQL CREATE TABLE statement
   * @param params Database DTO generation request parameters
   * @returns Generated DTO code as string
   */
  genDatabaseDto(params: GenDatabaseDtoRequest): Promise<string> {
    return request.post('/codegen/genDatabaseDto', params);
  },

  /**
   * Generate EF Code First models from database
   * @param params Code First generation request parameters
   * @returns EF generation output with code files
   */
  genCodeFirstFromDatabase(params: GenCodeFirstFromDatabaseRequest): Promise<EfGenerationOutput> {
    return request.post('/codegen/genCodeFirstFromDatabase', params);
  },

  /**
   * Generate proto code from gRPC client assembly
   * @param params Proto code generation request parameters
   * @returns Array of proto file information
   */
  genProtoCodeFromGrpcClientAssembly(params: GenProtoCodeFromGrpcClientAssemblyRequest): Promise<ProtoFileInfo[]> {
    const formData = new FormData();
    formData.append('namespaceName', params.namespaceName);
    const blob = new Blob([params.assembly as BlobPart], { type: 'application/octet-stream' });
    formData.append('assemblyFile', blob, 'assembly.dll');
    
    return request.post('/codegen/genProtoCodeFromGrpcClientAssembly', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  },
};