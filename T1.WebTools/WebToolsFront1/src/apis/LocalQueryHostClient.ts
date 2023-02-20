import createHttpClient from "@/apis/HttpClient";

export interface IUnbindLocalQueryAppInfo {
  appUid: string;
  port: number;
}

export interface IBindLocalQueryAppRequest {
  uniqueId: string;
  appUid: string;
}

export interface IBindLocalQueryAppResponse {
  errorMessage: string;
}

export class LocalQueryHostClient {
  _httpClient = createHttpClient(import.meta.env.VITE_LOCAL_QUERY_HOST_URL);

  getUnbindLocalQueryAppsAsync(): Promise<IUnbindLocalQueryAppInfo[]> {
    return this._httpClient.postAsync<IUnbindLocalQueryAppInfo[]>(
      "GetUnbindLocalQueryApps"
    );
  }

  async bindLocalQueryAppAsync(
    req: IBindLocalQueryAppRequest
  ): Promise<IBindLocalQueryAppResponse> {
    return await this._httpClient.postAsync<IBindLocalQueryAppResponse>(
      "bindLocalQueryApp",
      req
    );
  }
}

const client = new LocalQueryHostClient();
export default client;
