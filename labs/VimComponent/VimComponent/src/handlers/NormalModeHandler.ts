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
  
  addTMark(editor: any): void {
    const existingIndex = editor['tMarks'].findIndex(
      (mark: any) => mark.y === editor.cursorY && mark.x === editor.cursorX
    );
    
    if (existingIndex === -1) {
      editor['tMarks'].push({ y: editor.cursorY, x: editor.cursorX });
      editor['tMarks'].sort((a: any, b: any) => {
        if (a.y !== b.y) return a.y - b.y;
        return a.x - b.x;
      });
    }
  }
  
  clearTMarks(editor: any): void {
    editor['tMarks'] = [];
  }
}

