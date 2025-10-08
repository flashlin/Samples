import { EditorMode, BaseModeHandler } from '../vimEditorTypes';

export class MultiInsertModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.MultiInsert;
  
  onEnter(editor: any): void {
    editor['hiddenInput']?.focus();
  }
  
  onExit(editor: any): void {
    editor['hiddenInput']?.blur();
    editor['searchKeyword'] = '';
    editor['searchMatches'] = [];
    editor['currentMatchIndex'] = -1;
    editor['adjustCursorForNormalMode']();
  }
  
  shouldPreventDefault(key: string): boolean {
    return key.length !== 1;
  }
  
  handleKey(key: string, editor: any): void {
    if (key === 'Escape') {
      editor.mode = EditorMode.FastSearch;
      return;
    }
    
    if (key === 'Backspace') {
      this.multiInsertBackspace(editor);
      return;
    }
    
    if (key === 'Enter') {
      this.multiInsertNewline(editor);
      return;
    }
    
    if (key.length === 1) {
      this.multiInsertCharacter(editor, key);
    }
  }
  
  private multiInsertCharacter(editor: any, char: string): void {
    if (editor['currentMatchIndex'] < 0 || editor['searchMatches'].length === 0) {
      return;
    }
    
    const currentMatch = editor['searchMatches'][editor['currentMatchIndex']];
    const offsetInMatch = editor.cursorX - currentMatch.x;
    
    for (let i = editor['searchMatches'].length - 1; i >= 0; i--) {
      const match = editor['searchMatches'][i];
      const line = editor.content[match.y];
      const insertPos = match.x + offsetInMatch;
      
      editor.content[match.y] = 
        line.substring(0, insertPos) +
        char +
        line.substring(insertPos);
    }
    
    editor['searchKeyword'] = 
      editor['searchKeyword'].substring(0, offsetInMatch) + 
      char + 
      editor['searchKeyword'].substring(offsetInMatch);
    
    editor.cursorX++;
  }
  
  private multiInsertBackspace(editor: any): void {
    if (editor['currentMatchIndex'] < 0 || editor['searchMatches'].length === 0 || editor['searchKeyword'].length === 0) {
      return;
    }
    
    const currentMatch = editor['searchMatches'][editor['currentMatchIndex']];
    const offsetInMatch = editor.cursorX - currentMatch.x;
    
    if (offsetInMatch <= 0) {
      return;
    }
    
    for (let i = editor['searchMatches'].length - 1; i >= 0; i--) {
      const match = editor['searchMatches'][i];
      const line = editor.content[match.y];
      const deletePos = match.x + offsetInMatch - 1;
      
      editor.content[match.y] = 
        line.substring(0, deletePos) +
        line.substring(deletePos + 1);
    }
    
    editor['searchKeyword'] = 
      editor['searchKeyword'].substring(0, offsetInMatch - 1) + 
      editor['searchKeyword'].substring(offsetInMatch);
    
    editor.cursorX--;
  }
  
  private multiInsertNewline(editor: any): void {
    if (editor['currentMatchIndex'] < 0 || editor['searchMatches'].length === 0) {
      return;
    }
    
    const currentMatch = editor['searchMatches'][editor['currentMatchIndex']];
    const offsetInMatch = editor.cursorX - currentMatch.x;
    
    for (let i = editor['searchMatches'].length - 1; i >= 0; i--) {
      const match = editor['searchMatches'][i];
      const line = editor.content[match.y];
      const splitPos = match.x + offsetInMatch;
      
      const before = line.substring(0, splitPos);
      const after = line.substring(splitPos);
      
      editor.content[match.y] = before;
      editor.content.splice(match.y + 1, 0, after);
      
      for (let j = i + 1; j < editor['searchMatches'].length; j++) {
        if (editor['searchMatches'][j].y > match.y) {
          editor['searchMatches'][j].y++;
        }
      }
    }
    
    editor['searchKeyword'] = 
      editor['searchKeyword'].substring(0, offsetInMatch) + 
      '\n' + 
      editor['searchKeyword'].substring(offsetInMatch);
    
    editor.cursorY++;
    editor.cursorX = currentMatch.x;
  }
  
  handleInput(editor: any, value: string): void {
    for (const char of value) {
      this.multiInsertCharacter(editor, char);
    }
    if (editor['p5Instance']) {
      editor['p5Instance'].redraw();
    }
  }
  
  handleCompositionEnd(editor: any, data: string): void {
    if (data) {
      for (const char of data) {
        this.multiInsertCharacter(editor, char);
      }
      if (editor['p5Instance']) {
        editor['p5Instance'].redraw();
      }
    }
  }
}

