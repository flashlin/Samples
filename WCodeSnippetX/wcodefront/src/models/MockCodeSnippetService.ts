/** @format */

import {ICodeSnippetService, CodeSnippet} from './types';

export class MockCodeSnippetService implements ICodeSnippetService {
  _allItems = [
    new CodeSnippet({
      id: 1,
      programLanguage: 'cs',
      content: 'public class MyClass { }',
      description: 'This is a sample class',
    }),
    new CodeSnippet({
      id: 2,
      programLanguage: 'cs',
      content: 'public class User { }',
      description: 'This is a sample class',
    }),
    new CodeSnippet({
      id: 3,
      programLanguage: 'cs',
      content: 'public class cccc { }',
      description: 'This is a sample123 class',
    }),
  ];

  upsertCodeAsync(code: CodeSnippet): Promise<void> {
    if( code.id === 0 ){
      code.id = this._allItems.length + 1;
      this._allItems.push(code);
    } else {
      const index = this._allItems.findIndex(item => item.id === code.id);
      this._allItems[index] = code;
    }
    return Promise.resolve();
  }

  minimizeAsync(): Promise<void> {
    return Promise.resolve();
  }

  bringMeToFrontAsync(): Promise<void> {
    return Promise.resolve();
  }

  setClipboardAsync(text: string): Promise<void> {
    return new Promise(resolve => {
      navigator.clipboard.writeText(text);
      resolve();
    });
  }
  queryCodeAsync(text: string): Promise<CodeSnippet[]> {
    const data = this._allItems.filter(item => item.content.includes(text));
    return Promise.resolve(data);
  }
}
