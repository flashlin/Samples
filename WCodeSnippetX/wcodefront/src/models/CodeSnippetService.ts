import {ICodeSnippetService, CodeSnippet} from './types';

type BackendAn = (obj: IBoundObject) => Promise<void>;
type BackendFn = (obj: IBoundObject) => Promise<string>;

export class CodeSnippetService implements ICodeSnippetService {
  async setClipboardAsync(text: string): Promise<void> {
    return new Promise(resolve => {
      navigator.clipboard.writeText(text);
      resolve();
    });
  }

  async queryCodeAsync(text: string): Promise<CodeSnippet[]> {
    await CefSharp.BindObjectAsync('__backend');
    const resp = await window.__backend.queryCode(text);
    const data: CodeSnippet[] = JSON.parse(resp);
    return data;
  }

  minimizeAsync(): Promise<void> {
    return this.Invoke(backend => backend.minimize());
  }

  bringMeToFrontAsync(): Promise<void> {
    return this.Invoke(backend => backend.bringMeToFront());
  }

  async Invoke(callback: BackendAn): Promise<void> {
    await CefSharp.BindObjectAsync('__backend');
    return await callback(window.__backend);
  }

  async Call<T>(callback: BackendFn): Promise<T> {
    await CefSharp.BindObjectAsync('__backend');
    const resp = await callback(window.__backend);
    return JSON.parse(resp);
  }
}
