import createHttpClient from "@/apis/HttpClient";

export interface IUnbindLocalQueryAppInfo {
  appUid: string;
  port: number;
}

export interface IBindLocalQueryAppRequest {
  uniqueId: string;
  appUid: string;
}

export class LocalQueryHostClient {
  _httpClient = createHttpClient(import.meta.env.VUE_APP_LOCAL_QUERY_HOST_URL);

  getUnbindLocalQueryAppsAsync(): Promise<IUnbindLocalQueryAppInfo[]> {
    return this._httpClient.postAsync<IUnbindLocalQueryAppInfo[]>(
      "GetUnbindLocalQueryApps"
    );
  }

  async bindLocalQueryAppAsync(req: IBindLocalQueryAppRequest) {
    await this._httpClient.postVoidAsync("bindLocalQueryAppAsync", req);
  }
}

const client = new LocalQueryHostClient();
export default client;
