import createHttpClient from "@/apis/HttpClient";
import { MockAsyncMethod } from "../commons/MockUtils";

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

  @MockAsyncMethod([{ appUid: "xxx", port: 0 }])
  getUnbindLocalQueryAppsAsync(): Promise<IUnbindLocalQueryAppInfo[]> {
    return this._httpClient.postAsync<IUnbindLocalQueryAppInfo[]>(
      "GetUnbindLocalQueryApps"
    );
  }
}

const client = new LocalQueryHostClient();
export default client;
