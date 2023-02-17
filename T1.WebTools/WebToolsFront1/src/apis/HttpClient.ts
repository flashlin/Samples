import axios, { type AxiosInstance } from "axios";

export class HttpClient {
  _httpClient: AxiosInstance;
  constructor(baseUrl: string) {
    this._httpClient = axios.create({
      baseURL: baseUrl,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }
  async postAsync<T>(url: string, data?: unknown): Promise<T> {
    const response = await this._httpClient.post<T>(url, data);
    return response.data;
  }
}

function createHttpClient(baseUrl: string) {
  return new HttpClient(baseUrl);
}

export default createHttpClient;
