import createHttpClient from "@/apis/HttpClient";

export interface IKnockRequest {}

export interface IKnockResponse {
  appUid: string;
}

export interface IGetAllTableNamesResponse {
  tableNames: string[];
}

export class LocalQueryClient {
  _httpClient = createHttpClient("http://127.0.0.1/api/");

  knockAsync(req: IKnockRequest): Promise<IKnockResponse> {
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
