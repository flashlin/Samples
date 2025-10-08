import { EditorMode, BaseModeHandler } from '../vimEditorTypes';

export class VisualLineModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.VisualLine;
  
  handleKey(key: string, editor: any): void {
    if (editor['handleMovement'](key)) {
      return;
    }
    
    switch (key) {
      case 'Escape':
        editor.mode = EditorMode.Normal;
        break;
      case 'y':
        editor['yankVisualSelection']();
        editor.mode = EditorMode.Normal;
        break;
      case 'c':
      case 'd':
      case 'x':
        editor['cutVisualLineSelection']();
        editor.mode = EditorMode.Normal;
        break;
      case 'f':
        editor['previousMode'] = EditorMode.VisualLine;
        editor.mode = EditorMode.FastJump;
        editor['fastJumpMatches'] = [];
        editor['fastJumpInput'] = '';
        break;
    }
  }
}

