import createHttpClient, { HttpClient } from "@/apis/HttpClient";
import { MockAsyncMethod } from "../commons/MockUtils";

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

export interface ILocalQueryClient {
  knockAsync(req: IKnockRequest): Promise<IKnockResponse>;
  getAllTableNamesAsync(): Promise<IGetAllTableNamesResponse>;
}

export class LocalQueryClient implements ILocalQueryClient {
  _httpClient!: HttpClient;

  @MockAsyncMethod({ isSuccess: true })
  knockAsync(req: IKnockRequest): Promise<IKnockResponse> {
    this._httpClient = createHttpClient(
      `http://127.0.0.1:${req.port}/api/LocalQueryApi/`
    );
    return this._httpClient.postAsync<IKnockResponse>("knock", req);
  }

  @MockAsyncMethod({ tableNames: ["Customer", "Product"] })
  getAllTableNamesAsync(): Promise<IGetAllTableNamesResponse> {
    return this._httpClient.postAsync<IGetAllTableNamesResponse>(
      "getAllTableNames"
    );
  }
}

const client = new LocalQueryClient();
export default client;
