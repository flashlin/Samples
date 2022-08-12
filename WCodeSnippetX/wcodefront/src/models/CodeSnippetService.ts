import { ICodeSnippetService, CodeSnippet } from "./types";

export class CodeSnippetService implements ICodeSnippetService {
   async queryAsync(text: string): Promise<CodeSnippet[]> {
      await CefSharp.BindObjectAsync("__backend");
      let resp = await window.__backend.queryCode(text);
      let data: CodeSnippet[] = JSON.parse(resp);
      return data;
   }
}
