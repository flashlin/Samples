import axios, { type AxiosProgressEvent, type AxiosResponse } from "axios";

const decoder = new TextDecoder("utf-8");
export type GeneratingFn = (content: string) => void;
export function defaultGeneratingFn(content: string) {
   content
}

export interface IReadTokenStreamResponse {
   answer: string;
   lastToken: string;
}

export class JwtApi {
   private _apiUrl: string = "";
   private _urlPrefix: string = "/";

   constructor(apiUrl: string, urlPrefix: string) {
      this._apiUrl = apiUrl;
      this._urlPrefix = urlPrefix;
   }

   getUrl(relative_url: string): string {
      if (relative_url.startsWith("http:")) {
         return relative_url;
      }
      return `${this._apiUrl}${this._urlPrefix}${relative_url}`;
   }

   uploadFileAsync(url: string, file: File, onUploadProgress: (progressEvent: AxiosProgressEvent) => void) {
      const formData = new FormData();
      formData.append("file", file);
      return axios.post(this.getUrl(url),
         formData,
         {
            headers: {
               "Content-Type": "multipart/form-data"
            },
            onUploadProgress
         });
   }

   async getAsync<T>(url: string): Promise<T> {
      const accessToken = localStorage.getItem('accessToken');
      const resp = await axios
         .get(this.getUrl(url),
            {
               headers: { "Authorization": `Bearer ${accessToken}` }
            });
      return resp.data;
   }

   convertObjectToQueryString(paramsObject: any) {
      const urlSearchParams = new URLSearchParams(paramsObject as Record<string, string>);
      const queryString = urlSearchParams.toString();
      return queryString;
   }

   async postUrlAsync<T>(url: string, data: any = {}): Promise<T> {
      const queryString = this.convertObjectToQueryString(data);
      const accessToken = localStorage.getItem('accessToken');
      const resp = await axios
         .post(this.getUrl(url) + "?" + queryString,
            {
               headers: { "Authorization": `Bearer ${accessToken}` }
            });
      return resp.data;
   }

   async postAsync<T>(url: string, data: any = {}): Promise<T> {
      const accessToken = localStorage.getItem('accessToken');
      const resp = await axios.post(this.getUrl(url),
         JSON.stringify(data),
         {
            headers: {
               'Content-Type': 'application/json',
               Authorization: `Bearer ${accessToken}`,
            },
         });
      return resp.data;
   }

   async postResponseAsync(url: string, data: any = {}): Promise<AxiosResponse<any, any>> {
      const accessToken = localStorage.getItem('accessToken');
      const resp = await axios.post(this.getUrl(url),
         JSON.stringify(data),
         {
            headers: {
               'Content-Type': 'application/json',
               Authorization: `Bearer ${accessToken}`,
            },
         });
      return resp;
   }

   async postStreamAsync(url: string, data: any = {}) {
      const accessToken = localStorage.getItem('accessToken');
      const result = await fetch(this.getUrl(url), {
         method: "post",
         // signal: AbortSignal.timeout(8000),
         headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
         },
         body: JSON.stringify(data),
      });
      return result;
   }

   async postTokenStreamStringAsync(url: string, data: any = {}, generatingFn: GeneratingFn = defaultGeneratingFn): Promise<IReadTokenStreamResponse> {
      const { body, status } = await this.postStreamAsync(url, data);
      if (body) {
         const reader = body.getReader();
         return await this.readTokenStreamStringAsync(reader, status, generatingFn);
      }
      return {
         answer: "",
         lastToken: "",
      };
   }

   async readTokenStreamStringAsync(
      reader: ReadableStreamDefaultReader<Uint8Array>,
      status: number,
      generatingFn: GeneratingFn = defaultGeneratingFn
   ): Promise<IReadTokenStreamResponse> {
      let responseContent = "";
      let lastToken = "";
      while (status > 0) {
         const { value, done } = await reader.read();
         if (done) {
            break;
         }

         const decodedText = decoder.decode(value, { stream: true });
         if (status !== 200) {
            const error = JSON.parse(decodedText);
            throw new Error(error);
         }

         const chunks = decodedText.split(/\r?\n\r?\n/);
         for (const chunk of chunks) {
            if (chunk == '') {
               break;
            }
            const chunkData = chunk.replace(/^data: /, "");
            const content = JSON.parse(chunkData);
            if (content.isEnd) {
               lastToken = content.token;
               break;
            }
            generatingFn.call(generatingFn, content.token);
            responseContent += content.token;
         }
      }
      return {
         answer: responseContent,
         lastToken: lastToken
      };
   }

   async downloadStreamAsync(url: string, data: any, filename: string, downloadType: string = "application/jsonl") {
      const response = await this.postResponseAsync(url, data);
      const blob = new Blob([response.data], { type: downloadType });
      const blobUrl = window.URL.createObjectURL(blob);
      const aLink = document.createElement('a');
      aLink.href = blobUrl;
      aLink.download = filename;
      document.body.appendChild(aLink);
      aLink.click();
      document.body.removeChild(aLink);
      window.URL.revokeObjectURL(blobUrl);
   }

   async postFileAsync<T>(url: string, file: File, otherData: { [key: string]: string }): Promise<T> {
      const formData = new FormData();
      formData.append('File', file);
      for (const key in otherData) {
         if (otherData.hasOwnProperty(key)) {
            formData.append(key, otherData[key]);
         }
      }
      const resp = await axios.post(this.getUrl(url), formData, {
         headers: {
            'Content-Type': 'multipart/form-data',
         },
      });
      return resp.data;
   }
}

