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
      editor['multiInsertBackspace']();
      return;
    }
    
    if (key === 'Enter') {
      editor['multiInsertNewline']();
      return;
    }
    
    if (key.length === 1) {
      editor['multiInsertCharacter'](key);
    }
  }
}

