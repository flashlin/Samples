import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class InsertModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Insert;
  
  onEnter(editor: IVimEditor): void {
    editor['hiddenInput']?.focus();
  }
  
  onExit(editor: IVimEditor): void {
    editor['hiddenInput']?.blur();
    editor['adjustCursorForNormalMode']();
  }
  
  shouldPreventDefault(key: string): boolean {
    return key.length !== 1;
  }
  
  handleKey(key: string, editor: IVimEditor): void {
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
  
  handleInput(editor: IVimEditor, value: string): void {
    for (const char of value) {
      editor['insertCharacter'](char);
    }
    if (editor['p5Instance']) {
      editor['p5Instance'].redraw();
    }
  }
  
  handleCompositionEnd(editor: IVimEditor, data: string): void {
    if (data) {
      for (const char of data) {
        editor['insertCharacter'](char);
      }
      if (editor['p5Instance']) {
        editor['p5Instance'].redraw();
      }
    }
  }
}

