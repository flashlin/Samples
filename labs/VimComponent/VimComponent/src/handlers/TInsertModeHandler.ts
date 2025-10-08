import { EditorMode, BaseModeHandler } from '../vimEditorTypes';

export class TInsertModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.TInsert;
  
  onEnter(editor: any): void {
    editor['hiddenInput']?.focus();
  }
  
  onExit(editor: any): void {
    editor['hiddenInput']?.blur();
    editor['adjustCursorForNormalMode']();
  }
  
  shouldPreventDefault(key: string): boolean {
    return key.length !== 1;
  }
  
  handleKey(key: string, editor: any): void {
    if (key === 'Escape') {
      editor.mode = EditorMode.Normal;
      return;
    }
    
    if (key === 'Backspace') {
      editor['tInsertBackspace']();
    } else if (key === 'Enter') {
      editor['tInsertNewline']();
    } else if (key === 'ArrowLeft') {
      editor['moveCursorLeft']();
    } else if (key === 'ArrowRight') {
      editor['moveCursorRight']();
    } else if (key === 'ArrowUp') {
      editor['moveCursorUp']();
    } else if (key === 'ArrowDown') {
      editor['moveCursorDown']();
    }
  }
}

