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

export interface IVimEditor {
  mode: EditorMode;
  content: string[];
  cursorX: number;
  cursorY: number;
  
  // Public methods for getting ranges
  getInnerWordRange(): { startX: number; endX: number; y: number } | null;
  getInnerQuoteRange(quoteChar: string): { startY: number; startX: number; endY: number; endX: number } | null;
  getInnerBracketRange(): { startY: number; startX: number; endY: number; endX: number } | null;
  
  requestUpdate(property?: string, oldValue?: any): void;
  
  // Allow access to internal properties via index signature
  [key: string]: any;
}

export interface EditorModeHandler {
  readonly mode: EditorMode;
  handleKey(key: string, editor: IVimEditor): void;
  onEnter(editor: IVimEditor): void;
  onExit(editor: IVimEditor): void;
  shouldPreventDefault(key: string): boolean;
  handleInput?(editor: IVimEditor, value: string): void;
  handleCompositionEnd?(editor: IVimEditor, data: string): void;
}

export abstract class BaseModeHandler implements EditorModeHandler {
  abstract readonly mode: EditorMode;
  
  onEnter(editor: IVimEditor): void {}
  
  onExit(editor: IVimEditor): void {}
  
  shouldPreventDefault(key: string): boolean {
    return key.length !== 1;
  }
  
  handleInput(editor: IVimEditor, value: string): void {}
  
  handleCompositionEnd(editor: IVimEditor, data: string): void {}
  
  abstract handleKey(key: string, editor: IVimEditor): void;
}

