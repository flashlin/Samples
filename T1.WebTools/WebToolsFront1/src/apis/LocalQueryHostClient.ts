import createHttpClient from "@/apis/HttpClient";

class LocalQueryHostClient {
  _httpClient = createHttpClient(import.meta.env.VUE_APP_LOCAL_QUERY_HOST_URL);

  async knockAsync(req: IKnockRequest): Promise<IKnockResponse> {
    const resp = await this._httpClient.postAsync<IKnockResponse>("knock", req);
    return resp;
  }
}
