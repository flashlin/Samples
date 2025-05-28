import { JwtApi } from './jwtApi';

// Type for upload request
export interface UploadFileRequest {
  fileName: string;
  fileContent: Uint8Array; // JS 端用 Uint8Array 對應 C# byte[]
  offset: number;
}

// Type for upload response
export interface UploadFileResponse {
  fileName: string;
}

export class UploadFileApi {
  private jwtApi: JwtApi;

  constructor(apiUrl: string) {
    this.jwtApi = new JwtApi(apiUrl, '/api/');
  }

  /**
   * Upload file to server
   * @param req UploadFileRequest
   * @returns Promise<UploadFileResponse>
   */
  async upload(req: UploadFileRequest): Promise<UploadFileResponse> {
    const data = this.buildRequestBody(req);
    return await this.jwtApi.postAsync<UploadFileResponse>('/api/UploadFile/Upload', data);
  }

  /**
   * Build request body for upload
   */
  private buildRequestBody(req: UploadFileRequest): any {
    return {
      fileName: req.fileName,
      fileContent: Array.from(req.fileContent), // 轉成 number[] 傳給後端
      offset: req.offset,
    };
  }
} 