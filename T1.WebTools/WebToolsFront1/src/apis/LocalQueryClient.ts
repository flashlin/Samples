import createHttpClient from "@/apis/HttpClient";
import { HttpClient } from './HttpClient';

export interface IKnockRequest {
  uniqueId: string;
  appUid: string;
  port: number;
}

export interface IKnockResponse {
  isSuccess: boolean;
}

export interface IGetAllTableNamesResponse {
  tableNames: string[];
}

export class LocalQueryClient {
  _httpClient: HttpClient;

  knockAsync(req: IKnockRequest): Promise<IKnockResponse> {
    this._httpClient = createHttpClient(`http://127.0.0.1:${req.port}/api/LocalQueryApi/`);
    return this._httpClient.postAsync<IKnockResponse>("knock", req);
  }

  getAllTableNamesAsync(): Promise<IGetAllTableNamesResponse> {
    return this._httpClient.postAsync<IGetAllTableNamesResponse>(
      "getAllTableNames"
    );
  }
}

const client = new LocalQueryClient();
export default client;
