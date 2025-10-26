import { EditorMode, BaseModeHandler, IVimEditor, IntellisenseContext } from '../vimEditorTypes';

export class InsertModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Insert;
  
  onEnter(editor: IVimEditor): void {
    editor.hiddenInput?.focus();
  }
  
  onExit(editor: IVimEditor): void {
    const host = (editor as any).shadowRoot?.host as HTMLElement;
    if (host) {
      host.focus();
    }
    editor.adjustCursorForNormalMode();
  }
  
  shouldPreventDefault(key: string): boolean {
    return key.length !== 1;
  }
  
  triggerIntellisense(editor: IVimEditor): void {
    const content = (editor as any).content.join('\n');
    const cursorOffset = this.calculateCursorOffset(editor);
    const contentBeforeCursor = content.substring(0, cursorOffset);
    const contentAfterCursor = content.substring(cursorOffset);
    
    const cursorLine = editor.cursorY;
    const currentLineContent = (editor as any).content[cursorLine] || '';
    const cursorXInLine = editor.cursorX;
    const lineBeforeCursor = currentLineContent.substring(0, cursorXInLine);
    const lineAfterCursor = currentLineContent.substring(cursorXInLine);
    
    const context: IntellisenseContext = {
      mode: 'insert',
      contentBeforeCursor,
      contentAfterCursor,
      cursorLine,
      lineBeforeCursor,
      lineAfterCursor,
      cursorOffset
    };
    
    editor.emitIntellisense(context);
  }
  
  private calculateCursorOffset(editor: IVimEditor): number {
    const lines = (editor as any).content;
    let offset = 0;
    
    for (let i = 0; i < editor.cursorY; i++) {
      offset += lines[i].length + 1;
    }
    offset += editor.cursorX;
    
    return offset;
  }
  
  private getKeyPatterns(editor: IVimEditor) {
    // Patterns ordered by specificity
    return [
      { pattern: /^Escape$/, action: () => { editor.mode = EditorMode.Normal; } },
      { pattern: /^.$/, action: () => { /* Single character input handled by handleInput */ } },
      { pattern: /^Backspace$/, action: () => { editor.handleBackspace(); } },
      { pattern: /^Delete$/, action: () => { editor.handleDelete(); } },
      { pattern: /^Enter$/, action: () => { editor.handleEnter(); } },
      { pattern: /^ArrowLeft$/, action: () => { editor.moveCursorLeft(); } },
      { pattern: /^ArrowRight$/, action: () => { editor.moveCursorRight(); } },
      { pattern: /^ArrowUp$/, action: () => { editor.moveCursorUp(); } },
      { pattern: /^ArrowDown$/, action: () => { editor.moveCursorDown(); } },
    ];
  }
  
  handleKey(key: string, editor: IVimEditor): void {
    const keyPatterns = this.getKeyPatterns(editor);
    
    // Try to match patterns in order
    for (const { pattern, action } of keyPatterns) {
      if (pattern.test(key)) {
        action();
        return;
      }
    }
  }
  
  handleInput(editor: IVimEditor, value: string): void {
    for (const char of value) {
      editor.insertCharacter(char);
    }
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  handleCompositionEnd(editor: IVimEditor, data: string): void {
    if (data) {
      for (const char of data) {
        editor.insertCharacter(char);
      }
      if (editor.p5Instance) {
        editor.p5Instance.redraw();
      }
    }
  }
}

