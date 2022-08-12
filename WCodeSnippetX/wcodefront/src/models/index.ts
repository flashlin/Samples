export * from "./types";
import { MockCodeSnippetService } from "./MockCodeSnippetService";
import { CodeSnippetService } from "./CodeSnippetService";

export function useCodeSnippetService() {
   if( process.env.NODE_ENV === 'production' ){
      return new CodeSnippetService();
   }
   return new MockCodeSnippetService();
}
