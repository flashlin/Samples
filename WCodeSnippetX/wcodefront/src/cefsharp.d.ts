declare class CodeSnippet {
   constructor(data?: Partial<CodeSnippet>) {
      Object.assign(this, data);
   }
   id: number = 0;
   programLanguage: string = '';
   content: string = '';
   description: string = '';
}

declare interface ICodeSnippetService {
   getPort(): number;
   queryCode(text: string): Promise<string>;
}

declare interface Window {
   __backend: ICodeSnippetService;
}

declare var CefSharp: any;