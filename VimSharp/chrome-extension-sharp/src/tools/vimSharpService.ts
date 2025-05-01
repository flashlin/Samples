import { JwtApi } from "./jwtApi";

interface HookRequest {
}

interface JobInfo {
   id: string;
   name: string;
   method: string;
   parameters: string[];
}

interface JobResult {
   id: string;
   result: string;
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
   private _services: Map<string, any> = new Map();
   private _task: NodeJS.Timeout | null = null;

   constructor(urlPrefix: string = '/api/') {
      this._client = new JwtApi("http://127.0.0.1:8080", urlPrefix);
   }

   async startAsync(): Promise<void> {
      this._task = setInterval(async () => {
         try {
            const response = await this.getJobsInfoAsync();
            for (const job of response.jobs) {
               const service = this._services.get(job.name);
               if (service) {
                  const method = (service as any)[job.method];
                  if (typeof method === 'function') {
                     const result = await method.apply(service, job.parameters);
                     const jobResult: JobResult = {
                        id: job.id,
                        result: JSON.stringify(result)
                     };
                     await this.postJobResultAsync(jobResult);
                  }
               }
            }
         } catch (error) {
            console.error('Error processing jobs:', error);
         }
      }, 1000);
   }

   async stopAsync(): Promise<void> {
      if (this._task) {
         clearInterval(this._task);
         this._task = null;
      }
   }

   init() {
      const helloService = new HelloService();
      this._services.set('HelloService', helloService);
   }

   async postHookStreamStringAsync(apiUrl: string, req: HookRequest): Promise<void> {
      await this._client.postTokenStreamStringAsync(apiUrl, req, (_token: string) => {
      });
   }

   async getJobsInfoAsync(): Promise<GetJobsInfoResponse> {
      return await this._client.postAsync<GetJobsInfoResponse>("GetJobsInfo");
   }

   async postJobResultAsync(req: JobResult): Promise<void> {
      await this._client.postAsync("SendJobResult", req);
   }
}