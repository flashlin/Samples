import { EditorMode, BaseModeHandler } from '../vimEditorTypes';

export class VisualModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Visual;
  
  handleKey(key: string, editor: any): void {
    if (editor['visualKeyBuffer'] === '' && editor['handleMovement'](key)) {
      return;
    }
    
    if (key === 'Escape') {
      editor.mode = EditorMode.Normal;
      editor['visualKeyBuffer'] = '';
      return;
    }
    
    if (key === 'i' && editor['visualKeyBuffer'] === '') {
      editor['visualKeyBuffer'] = 'i';
      return;
    }
    
    if (editor['visualKeyBuffer'] === 'i' && (key === '`' || key === "'" || key === '"')) {
      editor['selectInnerQuote'](key);
      editor['visualKeyBuffer'] = '';
      return;
    }
    
    if (editor['visualKeyBuffer'] === 'i' && key === 'w') {
      editor['selectInnerWord']();
      editor['visualKeyBuffer'] = '';
      return;
    }
    
    editor['visualKeyBuffer'] = '';
    
    switch (key) {
      case 'y':
        editor['yankVisualSelection']();
        editor.mode = EditorMode.Normal;
        break;
      case 'c':
      case 'd':
      case 'x':
        editor['cutVisualSelection']();
        editor.mode = EditorMode.Normal;
        break;
      case 'f':
        editor['previousMode'] = EditorMode.Visual;
        editor.mode = EditorMode.FastJump;
        editor['fastJumpMatches'] = [];
        editor['fastJumpInput'] = '';
        break;
      case '*':
        editor['startSearchFromVisualSelection']();
        break;
    }
  }
}

