import { EditorMode, BaseModeHandler } from '../vimEditorTypes';

export class NormalModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Normal;
  
  handleKey(key: string, editor: any): void {
    if (editor['keyBuffer'] === '' && editor['handleMovement'](key)) {
      return;
    }
    
    editor['keyBuffer'] += key;
    editor['processKeyBuffer']();
  }
}

