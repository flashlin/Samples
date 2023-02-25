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
    [{ appUid: "08cbf4b7-10b6-4d6f-8ab6-212de3ba01af", port: 56426 }],
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
