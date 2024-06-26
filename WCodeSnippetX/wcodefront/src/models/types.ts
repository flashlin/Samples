/** @format */

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
  queryCodeAsync(text: string): Promise<CodeSnippet[]>;
  setClipboardAsync(text: string): Promise<void>;
  minimizeAsync(): Promise<void>;
  bringMeToFrontAsync(): Promise<void>;
  upsertCodeAsync(code: CodeSnippet): Promise<void>;
  deleteCodeAsync(id: number): Promise<void>;
}

export interface IAppState {
  selectedIndex: number;
  selectedItem: CodeSnippet;
  searchText: string;
  codeSnippetList: CodeSnippet[];
  isEditingData: boolean;
}
