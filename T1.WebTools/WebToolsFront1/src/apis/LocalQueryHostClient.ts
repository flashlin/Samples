import createHttpClient from "@/apis/HttpClient";
import { MockAsyncMethod } from "@/commons/MockUtils";

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

export interface ILocalQueryHostClient {
  getUnbindLocalQueryAppsAsync(): Promise<IUnbindLocalQueryAppInfo[]>;
}

export class LocalQueryHostClient implements ILocalQueryHostClient {
  _httpClient = createHttpClient(import.meta.env.VITE_LOCAL_QUERY_HOST_URL);

  @MockAsyncMethod(
    [{ appUid: "8482efb9-3cd3-4a22-84ab-5f9e21066b4b", port: 63569 }],
    true
  )
  getUnbindLocalQueryAppsAsync(): Promise<IUnbindLocalQueryAppInfo[]> {
    return this._httpClient.postAsync<IUnbindLocalQueryAppInfo[]>(
      "GetUnbindLocalQueryApps"
    );
  }
}

export default function useLocalQueryHostClient(): ILocalQueryHostClient {
  return new LocalQueryHostClient();
}
