import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class InsertModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Insert;
  
  onEnter(editor: IVimEditor): void {
    editor.hiddenInput?.focus();
  }
  
  onExit(editor: IVimEditor): void {
    editor.hiddenInput?.blur();
    editor.adjustCursorForNormalMode();
  }
  
  shouldPreventDefault(key: string): boolean {
    return key.length !== 1;
  }
  
  private getKeyPatterns(editor: IVimEditor) {
    // Patterns ordered by specificity
    return [
      { pattern: /^Escape$/, action: () => { editor.mode = EditorMode.Normal; } },
      { pattern: /^.$/, action: () => { /* Single character input handled by handleInput */ } },
      { pattern: /^Backspace$/, action: () => { editor.handleBackspace(); } },
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

