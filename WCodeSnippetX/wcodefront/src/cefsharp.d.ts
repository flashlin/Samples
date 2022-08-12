declare interface ICodeSnippetService {
   queryCode(text: string): Promise<string>;
}

declare interface Window {
   __backend: ICodeSnippetService;
}

declare var CefSharp: any;