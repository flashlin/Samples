import { JwtApi } from "./jwtApi";

interface HookRequest {
}

interface JobInfo {
   id: string;
   name: string;
   parameters: string[];
}

interface GetJobsInfoResponse {
   jobs: JobInfo[];
}

export class HelloService {
   async getHelloAsync(): Promise<string> {
      return Promise.resolve("Hello");
   }
}

export class VimSharpService {
   private _client: JwtApi;

   constructor(urlPrefix: string = '/api/') {
      this._client = new JwtApi("http://127.0.0.1:8080", urlPrefix);
   }

   async startAsync(): Promise<void> {
      
   }

   async stopAsync(): Promise<void> {
   }

   init() {

   }

   async postHookStreamStringAsync(apiUrl: string, req: HookRequest): Promise<void> {
      await this._client.postTokenStreamStringAsync(apiUrl, req, (token: string) => {
      });
   }

   async getJobsInfoAsync(): Promise<GetJobsInfoResponse> {
      return await this._client.postAsync<GetJobsInfoResponse>("GetJobsInfo");
   }
}