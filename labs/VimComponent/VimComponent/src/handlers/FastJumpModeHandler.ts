import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class FastJumpModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.FastJump;
  
  handleKey(key: string, editor: IVimEditor): void {
    if (key === 'Escape') {
      editor.mode = editor.previousMode;
      editor.fastJumpMatches = [];
      editor.fastJumpInput = '';
      return;
    }
    
    if (key.length === 1) {
      const matches = editor.findMatchesInVisibleRange(key);
      
      if (matches.length === 0) {
        editor.mode = editor.previousMode;
        editor.fastJumpMatches = [];
        editor.fastJumpInput = '';
      } else if (matches.length === 1) {
        editor.cursorX = matches[0].x;
        editor.cursorY = matches[0].y;
        editor.mode = editor.previousMode;
        editor.fastJumpMatches = [];
        editor.fastJumpInput = '';
      } else {
        editor.fastJumpMatches = matches;
        editor.mode = EditorMode.FastMatch;
      }
    }
  }
}

