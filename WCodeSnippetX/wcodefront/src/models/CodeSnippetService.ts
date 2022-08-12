import {ICodeSnippetService, CodeSnippet} from './types';

export class CodeSnippetService implements ICodeSnippetService {
  async setClipboardAsync(text: string): Promise<void> {
    //await CefSharp.BindObjectAsync("__backend");
    //await window.__backend.setClipboardAsync(text);
    return new Promise(resolve => {
      navigator.clipboard.writeText(text);
      resolve();
    });
  }
  async queryCodeAsync(text: string): Promise<CodeSnippet[]> {
    await CefSharp.BindObjectAsync('__backend');
    const resp = await window.__backend.queryCodeAsync(text);
    const data: CodeSnippet[] = JSON.parse(resp);
    return data;
  }
}
