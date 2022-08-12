import { ICodeSnippetService, CodeSnippet } from "./types";

export class MockCodeSnippetService implements ICodeSnippetService {
   queryAsync(text: string): Promise<CodeSnippet[]> {
      return Promise.resolve([
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
      ]);
   }
}
