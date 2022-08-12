export * from "./types";
import { MockCodeSnippetService } from "./MockCodeSnippetService";
import { CodeSnippetService } from "./CodeSnippetService";

export function useCodeSnippetService() {
   return new MockCodeSnippetService();
   //return new CodeSnippetService();
}
