import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class VisualLineModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.VisualLine;
  
  shouldPreventDefault(key: string): boolean {
    return true;
  }
  
  async handleKey(key: string, editor: IVimEditor): Promise<void> {
    if (editor.handleMovement(key)) {
      return;
    }
    
    switch (key) {
      case 'Escape':
        editor.mode = EditorMode.Normal;
        break;
      case 'y':
        await this.yankVisualSelection(editor);
        editor.mode = EditorMode.Normal;
        break;
      case 'c':
      case 'd':
      case 'x':
        await this.cutVisualLineSelection(editor);
        editor.mode = EditorMode.Normal;
        break;
      case 'f':
        editor.previousMode = EditorMode.VisualLine;
        editor.mode = EditorMode.FastJump;
        editor.fastJumpMatches = [];
        editor.fastJumpInput = '';
        break;
    }
  }
  
  private getVisualSelection(editor: IVimEditor): string {
    const startY = Math.min(editor.visualStartY, editor.cursorY);
    const endY = Math.max(editor.visualStartY, editor.cursorY);
    
    // Visual Line Mode: always select entire lines
    const lines: string[] = [];
    for (let y = startY; y <= endY; y++) {
      lines.push(editor.content[y]);
    }
    
    return lines.join('\n');
  }
  
  private async yankVisualSelection(editor: IVimEditor): Promise<void> {
    const selection = this.getVisualSelection(editor);
    await editor.copyToClipboard(selection, true);
  }
  
  private async cutVisualLineSelection(editor: IVimEditor): Promise<void> {
    editor.saveHistory();
    
    const selection = this.getVisualSelection(editor);
    await editor.copyToClipboard(selection, true);
    
    const startY = Math.min(editor.visualStartY, editor.cursorY);
    const endY = Math.max(editor.visualStartY, editor.cursorY);
    const linesToDelete = endY - startY + 1;
    
    editor.content.splice(startY, linesToDelete);
    
    if (editor.content.length === 0) {
      editor.content = [''];
    }
    
    editor.cursorY = Math.min(startY, editor.content.length - 1);
    editor.cursorX = 0;
    editor.adjustCursorX();
  }
}

