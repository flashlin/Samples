export class CodeSnippet {
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