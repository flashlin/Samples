import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class FastMatchModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.FastMatch;
  
  handleKey(key: string, editor: IVimEditor): void {
    if (key === 'Escape') {
      editor.mode = editor['previousMode'];
      editor['fastJumpMatches'] = [];
      editor['fastJumpInput'] = '';
      return;
    }
    
    if (key.length === 1 && /[a-z]/.test(key)) {
      editor['fastJumpInput'] += key;
      
      const matchingLabels = editor['fastJumpMatches'].filter((match: any) => 
        match.label.startsWith(editor['fastJumpInput'])
      );
      
      if (matchingLabels.length === 0) {
        editor.mode = editor['previousMode'];
        editor['fastJumpMatches'] = [];
        editor['fastJumpInput'] = '';
      } else if (matchingLabels.length === 1 && matchingLabels[0].label === editor['fastJumpInput']) {
        editor.cursorX = matchingLabels[0].x;
        editor.cursorY = matchingLabels[0].y;
        editor.mode = editor['previousMode'];
        editor['fastJumpMatches'] = [];
        editor['fastJumpInput'] = '';
      } else {
        editor['fastJumpMatches'] = matchingLabels;
      }
    }
  }
}

