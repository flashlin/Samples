export class CodeSnippet {
   constructor(data?: Partial<CodeSnippet>) {
      Object.assign(this, data);
   }
   id: number = 0;
   programLanguage: string = '';
   content: string = '';
   description: string = '';
}

export class CodeSnippetService {
   query(text: string): CodeSnippet[] {
      return [
         new CodeSnippet({
            id: 1,
            programLanguage: 'cs',
            content: 'public class MyClass { }',
            description: 'This is a sample class'
         }),
         new CodeSnippet({
            id: 2,
            programLanguage: 'cs',
            content: 'public class User { }',
            description: 'This is a sample class'
         }),
         new CodeSnippet({
            id: 3,
            programLanguage: 'cs',
            content: 'public class cccc { }',
            description: 'This is a sample123 class'
         }),
      ];
   }
}