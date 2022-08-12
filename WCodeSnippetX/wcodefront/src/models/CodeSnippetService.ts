import { ICodeSnippetService, CodeSnippet } from "./types";

export class CodeSnippetService implements ICodeSnippetService {
   async queryAsync(text: string): Promise<CodeSnippet[]> {
      await CefSharp.BindObjectAsync("__backend");
      const resp = await window.__backend.queryCode(text);
      const data: CodeSnippet[] = JSON.parse(resp);
      return data;
   }
}
