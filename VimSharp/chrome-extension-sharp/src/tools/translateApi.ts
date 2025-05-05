import { GptApi } from './gptApi';

const GPTAPI_URL = import.meta.env.VITE_GPTAPI_URL;

export enum ModelType {
   Ollama,
   Custom
}

// GptReq interface
export interface GptReq {
  modelType: ModelType;
  question: string;
  temperature: number;
  maxTokens: number;
}

export async function invokeGptApiAsync(text: string) {
  const gptApi = new GptApi(GPTAPI_URL);
  const req: GptReq = {
     modelType: ModelType.Custom,
     question: text,
     temperature: 0.5,
     maxTokens: 1000
  };
  const cursorChar = "▌";
  let output = "";
  await gptApi.postTokenStreamStringAsync('MessageStream', req, (token: string) => {
      let outputText = output;
      let answer = outputText.substring(0, outputText.lastIndexOf(cursorChar));
      output = answer + token + cursorChar;
  });
  return output;
}

// Translate to English using GPT API
export async function translateToEnAsync(text: string) {
  // Create GPT API instance
  const prompt = `Translate the following text to English:\n${text}`;
  return await invokeGptApiAsync(prompt);
}

export async function translateToZhAsync(text: string): Promise<string> {
  const prompt = `Translate the following text to Traditional Chinese:\n${text}`;
  return await invokeGptApiAsync(prompt);
}