import createHttpClient from "@/apis/HttpClient";

export interface IUnbindLocalQueryAppInfo {
  appUid: string;
  port: number;
}

export class LocalQueryHostClient {
  _httpClient = createHttpClient(import.meta.env.VUE_APP_LOCAL_QUERY_HOST_URL);

  getUnbindLocalQueryAppsAsync(): Promise<IUnbindLocalQueryAppInfo[]> {
    return this._httpClient.postAsync<IUnbindLocalQueryAppInfo[]>(
      "GetUnbindLocalQueryApps"
    );
  }
}

const client = new LocalQueryHostClient();
export default client;
