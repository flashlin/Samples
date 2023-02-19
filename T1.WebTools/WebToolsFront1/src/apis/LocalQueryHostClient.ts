import createHttpClient from "@/apis/HttpClient";

export class LocalQueryHostClient {
  _httpClient = createHttpClient(import.meta.env.VUE_APP_LOCAL_QUERY_HOST_URL);
}

const client = new LocalQueryHostClient();
export default client;
