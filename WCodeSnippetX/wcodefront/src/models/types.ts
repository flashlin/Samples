export class CodeSnippet {
   static Empty = new CodeSnippet({
      id: 0,   
   });
   constructor(data?: Partial<CodeSnippet>) {
      Object.assign(this, data);
   }
   id: number = 0;
   programLanguage: string = '';
   content: string = '';
   description: string = '';
}

export interface ICodeSnippetService {
   queryAsync(text: string): Promise<CodeSnippet[]>;
}

export interface IAppState {
   selectedIndex: number;
   selectedItem: CodeSnippet;
   searchText: string;
   codeSnippetList: CodeSnippet[];
   filterCodes: CodeSnippet[];
}