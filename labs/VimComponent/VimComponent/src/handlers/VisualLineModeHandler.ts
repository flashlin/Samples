import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class VisualLineModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.VisualLine;
  
  handleKey(key: string, editor: IVimEditor): void {
    if (editor['handleMovement'](key)) {
      return;
    }
    
    switch (key) {
      case 'Escape':
        editor.mode = EditorMode.Normal;
        break;
      case 'y':
        this.yankVisualSelection(editor);
        editor.mode = EditorMode.Normal;
        break;
      case 'c':
      case 'd':
      case 'x':
        this.cutVisualLineSelection(editor);
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
  
  private getVisualSelection(editor: IVimEditor): string {
    const startY = Math.min(editor['visualStartY'], editor.cursorY);
    const endY = Math.max(editor['visualStartY'], editor.cursorY);
    const startX = editor['visualStartY'] === startY ? 
      Math.min(editor['visualStartX'], editor.cursorX) : 
      Math.min(editor.cursorX, editor['visualStartX']);
    const endX = editor['visualStartY'] === endY ? 
      Math.max(editor['visualStartX'], editor.cursorX) : 
      Math.max(editor.cursorX, editor['visualStartX']);
    
    if (startY === endY) {
      return editor.content[startY].slice(startX, endX + 1);
    }
    
    let result = '';
    for (let y = startY; y <= endY; y++) {
      if (y === startY) {
        result += editor.content[y].slice(startX) + '\n';
      } else if (y === endY) {
        result += editor.content[y].slice(0, endX + 1);
      } else {
        result += editor.content[y] + '\n';
      }
    }
    return result;
  }
  
  private yankVisualSelection(editor: IVimEditor): void {
    const selection = this.getVisualSelection(editor);
    navigator.clipboard.writeText(selection);
  }
  
  private cutVisualLineSelection(editor: IVimEditor): void {
    editor['saveHistory']();
    
    const selection = this.getVisualSelection(editor);
    navigator.clipboard.writeText(selection);
    
    const startY = Math.min(editor['visualStartY'], editor.cursorY);
    const endY = Math.max(editor['visualStartY'], editor.cursorY);
    const linesToDelete = endY - startY + 1;
    
    editor.content.splice(startY, linesToDelete);
    
    if (editor.content.length === 0) {
      editor.content = [''];
    }
    
    editor.cursorY = Math.min(startY, editor.content.length - 1);
    editor.cursorX = 0;
    editor['adjustCursorX']();
  }
}

