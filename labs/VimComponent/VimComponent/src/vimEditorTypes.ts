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
  TVisual = 't-visual',
  Command = 'command',
  SearchInput = 'search-input',
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

export interface TextRange {
  startY: number;
  startX: number;
  endY: number;
  endX: number;
}

export interface IVimEditor {
  // State properties
  mode: EditorMode;
  content: string[];
  cursorX: number;
  cursorY: number;
  fastJumpMatches: Array<{ x: number; y: number; label: string }>;
  fastJumpInput: string;
  keyBuffer: string;
  previousMode: EditorMode.Normal | EditorMode.Visual | EditorMode.VisualLine;
  visualStartX: number;
  visualStartY: number;
  searchKeyword: string;
  searchMatches: Array<{ y: number; x: number }>;
  currentMatchIndex: number;
  searchHistory: Array<{ keyword: string; matches: Array<{ y: number; x: number }> }>;
  tMarks: Array<{ y: number; x: number }>;
  commandInput: string;
  searchInput: string;
  hiddenInput: HTMLInputElement | null;
  p5Instance: any;
  modeHandlerRegistry: any;
  
  // Range methods
  getInnerWordRange(): { startX: number; endX: number; y: number } | null;
  getInnerQuoteRange(quoteChar: string): TextRange | null;
  getInnerBracketRange(): TextRange | null;
  findInnerBracketRange(openChar: string, closeChar: string): TextRange | null;
  
  // Movement methods
  handleMovement(key: string): boolean;
  moveCursorUp(): void;
  moveCursorDown(): void;
  moveCursorLeft(): void;
  moveCursorRight(): void;
  moveToFirstLine(): void;
  jumpToMatchingBracket(): void;
  
  // Edit methods
  insertCharacter(char: string): void;
  handleEnter(): void;
  handleBackspace(): void;
  deleteWord(): void;
  deleteToWordEnd(): void;
  deleteToLineEnd(): void;
  deleteLine(): void;
  deleteLinesDown(count: number): void;
  deleteLinesUp(count: number): void;
  deleteMultiLineSelection(startY: number, endY: number, startX: number, endX: number): void;
  pasteAfterCursor(): void;
  pasteBeforeCursor(): void;
  insertLineBelow(): void;
  
  // Mode methods
  enterInsertMode(): void;
  
  // History methods
  saveHistory(cursorPos?: { cursorX: number; cursorY: number }): void;
  undo(): void;
  
  // Helper methods
  adjustCursorX(): void;
  adjustCursorForNormalMode(): void;
  updateInputPosition(): void;
  findMatchesInVisibleRange(char: string): Array<{ x: number; y: number; label: string }>;
  
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

export interface IntellisenseContext {
  mode: string;
  contentBeforeCursor: string;
  contentAfterCursor: string;
  cursorLine: number;
  lineBeforeCursor: string;
  lineAfterCursor: string;
  cursorOffset: number;
}

export interface IntellisenseItem {
  text: string;
  description?: string;
  action: () => void;
}

export interface KeyPressEventDetail {
  key: string;
  mode: EditorMode;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
  metaKey: boolean;
  cursorX: number;
  cursorY: number;
}

export interface ChangeEventDetail {
  content: string[];
}

export interface CommandEventDetail {
  command: string;
}

