// gptApi.ts

// Type for the generating function
export type GeneratingFn = (token: string) => void;

// Default generating function (does nothing)
export const defaultGeneratingFn: GeneratingFn = (_ :string) => {};

// Type for the response of reading token stream
export interface IReadTokenStreamResponse {
  answer: string;
  lastToken: string;
}

// Utility for decoding Uint8Array to string
const decoder = new TextDecoder();

// Main API class
export class GptApi {
  private _apiUrl: string = "";
  private _urlPrefix: string = "/";

  constructor(apiUrl: string, urlPrefix: string="/api/") {
     this._apiUrl = apiUrl;
     this._urlPrefix = urlPrefix;
  }

  // Get full URL
  getUrl(relative_url: string): string {
    if (relative_url.startsWith("http:")) {
       return relative_url;
    }
    return `${this._apiUrl}${this._urlPrefix}${relative_url}`;
 }

  // Post request and get stream response
  async postStreamAsync(url: string, data: any = {}): Promise<Response> {
    const accessToken = localStorage.getItem('accessToken');
    const result = await fetch(this.getUrl(url), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(data),
    });
    return result;
  }

  // Post and process token stream as string
  async postTokenStreamStringAsync(
    url: string,
    data: any = {},
    generatingFn: GeneratingFn = defaultGeneratingFn
  ): Promise<IReadTokenStreamResponse> {
    const result = await this.postStreamAsync(url, data);
    const { body, status } = result;
    if (body) {
      const reader = body.getReader();
      return await this.readTokenStreamStringAsync(reader, status, generatingFn);
    }
    return {
      answer: '',
      lastToken: '',
    };
  }

  // Read and process the token stream
  async readTokenStreamStringAsync(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    status: number,
    generatingFn: GeneratingFn = defaultGeneratingFn
  ): Promise<IReadTokenStreamResponse> {
    let responseContent = '';
    let lastToken = '';
    while (status > 0) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      // Decode chunk
      const decodedText = decoder.decode(value, { stream: true });
      if (status !== 200) {
        const error = JSON.parse(decodedText);
        throw new Error(error);
      }
      // Split by double newlines
      const chunks = decodedText.split(/\r?\n\r?\n/);
      for (const chunk of chunks) {
        if (chunk === '') {
          break;
        }
        const chunkData = chunk.replace(/^data: /, '');
        const content = JSON.parse(chunkData);
        if (content.isEnd) {
          lastToken = content.token;
          break;
        }
        generatingFn(content.token);
        responseContent += content.token;
      }
    }
    return {
      answer: responseContent,
      lastToken: lastToken,
    };
  }
} 