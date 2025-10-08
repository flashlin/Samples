import { EditorMode, BaseModeHandler } from '../vimEditorTypes';

export class InsertModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Insert;
  
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
    
    if (key.length === 1) {
      return;
    }
    
    if (key === 'Backspace') {
      editor['handleBackspace']();
    } else if (key === 'Enter') {
      editor['handleEnter']();
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

