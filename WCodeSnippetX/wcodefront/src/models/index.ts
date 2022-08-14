export * from "./types";
import { MockCodeSnippetService } from "./MockCodeSnippetService";
import { CodeSnippetService } from "./CodeSnippetService";

const codeSnippetService = process.env.NODE_ENV === 'production' ?
   new CodeSnippetService() : new MockCodeSnippetService();

export function useCodeSnippetService() {
   return codeSnippetService;
}
