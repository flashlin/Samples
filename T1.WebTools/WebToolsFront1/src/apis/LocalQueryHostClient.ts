import createHttpClient from "@/apis/HttpClient";

class LocalQueryHostClient {
  _httpClient = createHttpClient(import.meta.env.VUE_APP_LOCAL_QUERY_HOST_URL);
}
