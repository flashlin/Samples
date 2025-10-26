import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class TInsertModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.TInsert;
  
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
    return true;
  }
  
  private getKeyPatterns(editor: IVimEditor) {
    // Patterns ordered by specificity
    return [
      { pattern: /^Escape$/, action: () => { editor.mode = EditorMode.Normal; } },
      { pattern: /^Backspace$/, action: () => { this.tInsertBackspace(editor); } },
      { pattern: /^Enter$/, action: () => { this.tInsertNewline(editor); } },
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
  
  tInsertBackspace(editor: IVimEditor): void {
    if (editor.tMarks.length === 0) return;
    
    editor.saveHistory();
    
    const currentMarkIndex = this.findCurrentTMarkIndex(editor);
    if (currentMarkIndex === -1) return;
    
    const currentMark = editor.tMarks[currentMarkIndex];
    const offsetInMark = editor.cursorX - currentMark.x;
    
    if (offsetInMark <= 0) {
      return;
    }
    
    for (let i = editor.tMarks.length - 1; i >= 0; i--) {
      const mark = editor.tMarks[i];
      const line = editor.content[mark.y];
      const deletePos = mark.x + offsetInMark - 1;
      
      if (deletePos >= 0 && deletePos < line.length) {
        editor.content[mark.y] = line.substring(0, deletePos) + line.substring(deletePos + 1);
      }
    }
    
    editor.cursorX--;
  }
  
  tInsertNewline(editor: IVimEditor): void {
    if (editor.tMarks.length === 0) return;
    
    editor.saveHistory();
    
    const currentMarkIndex = this.findCurrentTMarkIndex(editor);
    if (currentMarkIndex === -1) return;
    
    const currentMark = editor.tMarks[currentMarkIndex];
    const offsetInMark = editor.cursorX - currentMark.x;
    
    for (let i = editor.tMarks.length - 1; i >= 0; i--) {
      const mark = editor.tMarks[i];
      const line = editor.content[mark.y];
      const splitPos = mark.x + offsetInMark;
      
      const before = line.substring(0, splitPos);
      const after = line.substring(splitPos);
      
      editor.content[mark.y] = before;
      editor.content.splice(mark.y + 1, 0, after);
      
      for (let j = i + 1; j < editor.tMarks.length; j++) {
        if (editor.tMarks[j].y > mark.y) {
          editor.tMarks[j].y++;
        }
      }
    }
    
    editor.cursorY++;
    editor.cursorX = currentMark.x;
  }
  
  tInsertCharacter(editor: IVimEditor, char: string): void {
    if (editor.tMarks.length === 0) return;
    
    const currentMarkIndex = this.findCurrentTMarkIndex(editor);
    if (currentMarkIndex === -1) return;
    
    const currentMark = editor.tMarks[currentMarkIndex];
    const offsetInMark = editor.cursorX - currentMark.x;
    
    for (let i = editor.tMarks.length - 1; i >= 0; i--) {
      const mark = editor.tMarks[i];
      const line = editor.content[mark.y];
      const insertPos = mark.x + offsetInMark;
      
      editor.content[mark.y] = 
        line.substring(0, insertPos) +
        char +
        line.substring(insertPos);
    }
    
    editor.cursorX++;
  }
  
  private findCurrentTMarkIndex(editor: IVimEditor): number {
    if (editor.tMarks.length === 0) return -1;
    
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      if (mark.y === editor.cursorY && mark.x === editor.cursorX) {
        return i;
      }
    }
    
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      if (mark.y === editor.cursorY && editor.cursorX >= mark.x) {
        const nextMark = editor.tMarks[i + 1];
        if (!nextMark || nextMark.y !== editor.cursorY || editor.cursorX < nextMark.x) {
          return i;
        }
      }
    }
    
    for (let i = editor.tMarks.length - 1; i >= 0; i--) {
      const mark = editor.tMarks[i];
      if (mark.y < editor.cursorY || (mark.y === editor.cursorY && mark.x <= editor.cursorX)) {
        return i;
      }
    }
    
    return 0;
  }
  
  handleInput(editor: IVimEditor, value: string): void {
    for (const char of value) {
      this.tInsertCharacter(editor, char);
    }
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  handleCompositionEnd(editor: IVimEditor, data: string): void {
    if (data) {
      for (const char of data) {
        this.tInsertCharacter(editor, char);
      }
      if (editor.p5Instance) {
        editor.p5Instance.redraw();
      }
    }
  }
}

