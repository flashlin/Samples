export enum EditorMode {
  Normal = 'normal',
  Insert = 'insert',
  Visual = 'visual',
  VisualLine = 'visual-line',
  FastJump = 'fast-jump',
  FastMatch = 'fast-match',
  FastSearch = 'fast-search',
  MultiInsert = 'multi-insert',
  TInsert = 't-insert',
}

export interface EditorStatus {
  mode: EditorMode;
  cursorX: number;
  cursorY: number;
  cursorVisible: boolean;
  searchKeyword?: string;
  searchMatchCount?: number;
}

export interface BufferCell {
  char: string;
  foreground: number[];
  background: number[];
}

export interface EditorModeHandler {
  readonly mode: EditorMode;
  handleKey(key: string, editor: any): void;
  onEnter(editor: any): void;
  onExit(editor: any): void;
  shouldPreventDefault(key: string): boolean;
}

export abstract class BaseModeHandler implements EditorModeHandler {
  abstract readonly mode: EditorMode;
  
  onEnter(editor: any): void {}
  
  onExit(editor: any): void {}
  
  shouldPreventDefault(key: string): boolean {
    return key.length !== 1;
  }
  
  abstract handleKey(key: string, editor: any): void;
}

