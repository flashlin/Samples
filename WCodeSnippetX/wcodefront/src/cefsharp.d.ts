declare interface ICodeSnippetService {
  queryCode(text: string): Promise<string>;
}

declare interface Window {
  __backend: ICodeSnippetService;
}

declare let CefSharp: any;
