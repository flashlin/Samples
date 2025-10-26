import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class VisualModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Visual;
  private visualKeyBuffer = '';
  
  shouldPreventDefault(key: string): boolean {
    return true;
  }
  
  async handleKey(key: string, editor: IVimEditor): Promise<void> {
    if (this.visualKeyBuffer === '' && editor.handleMovement(key)) {
      return;
    }
    
    if (key === 'Escape') {
      editor.mode = EditorMode.Normal;
      this.visualKeyBuffer = '';
      return;
    }
    
    if (key === 'i' && this.visualKeyBuffer === '') {
      this.visualKeyBuffer = 'i';
      return;
    }
    
    if (this.visualKeyBuffer === 'i' && (key === '`' || key === "'" || key === '"')) {
      this.selectInnerQuote(editor, key);
      this.visualKeyBuffer = '';
      return;
    }
    
    if (this.visualKeyBuffer === 'i' && key === 'w') {
      this.selectInnerWord(editor);
      this.visualKeyBuffer = '';
      return;
    }
    
    this.visualKeyBuffer = '';
    
    switch (key) {
      case 'y':
        await this.yankVisualSelection(editor);
        editor.mode = EditorMode.Normal;
        break;
      case 'c':
      case 'd':
      case 'x':
        await this.cutVisualSelection(editor);
        editor.mode = EditorMode.Normal;
        break;
      case 'f':
        editor.previousMode = EditorMode.Visual;
        editor.mode = EditorMode.FastJump;
        editor.fastJumpMatches = [];
        editor.fastJumpInput = '';
        break;
      case '*':
        this.startSearchFromVisualSelection(editor);
        break;
    }
  }
  
  private selectInnerWord(editor: IVimEditor): void {
    const range = editor.getInnerWordRange();
    if (!range) {
      editor.mode = EditorMode.Normal;
      return;
    }
    
    editor.mode = EditorMode.Visual;
    editor.visualStartY = range.y;
    editor.visualStartX = range.startX;
    editor.cursorY = range.y;
    editor.cursorX = range.endX;
    
    editor.updateInputPosition();
  }
  
  private selectInnerQuote(editor: IVimEditor, quoteChar: string): void {
    const range = editor.getInnerQuoteRange(quoteChar);
    if (!range) {
      return;
    }
    
    editor.visualStartY = range.startY;
    editor.visualStartX = range.startX + 1;
    editor.cursorY = range.endY;
    editor.cursorX = range.endX - 1;
    
    editor.updateInputPosition();
  }
  
  private getVisualSelection(editor: IVimEditor): string {
    const startY = Math.min(editor.visualStartY, editor.cursorY);
    const endY = Math.max(editor.visualStartY, editor.cursorY);
    const startX = editor.visualStartY === startY ? 
      Math.min(editor.visualStartX, editor.cursorX) : 
      Math.min(editor.cursorX, editor.visualStartX);
    const endX = editor.visualStartY === endY ? 
      Math.max(editor.visualStartX, editor.cursorX) : 
      Math.max(editor.cursorX, editor.visualStartX);
    
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
  
  private async yankVisualSelection(editor: IVimEditor): Promise<void> {
    const selection = this.getVisualSelection(editor);
    await editor.copyToClipboard(selection, false);
  }
  
  private async cutVisualSelection(editor: IVimEditor): Promise<void> {
    const selection = this.getVisualSelection(editor);
    await editor.copyToClipboard(selection, false);
    
    const startY = Math.min(editor.visualStartY, editor.cursorY);
    const endY = Math.max(editor.visualStartY, editor.cursorY);
    
    let startX, endX;
    if (startY === endY) {
      startX = Math.min(editor.visualStartX, editor.cursorX);
      endX = Math.max(editor.visualStartX, editor.cursorX);
    } else {
      if (editor.visualStartY === startY) {
        startX = editor.visualStartX;
        endX = editor.cursorX;
      } else {
        startX = editor.cursorX;
        endX = editor.visualStartX;
      }
    }

    editor.saveHistory({ cursorX: endX, cursorY: endY });
    
    editor.deleteMultiLineSelection(startY, endY, startX, endX);
    editor.adjustCursorX();
  }
  
  private startSearchFromVisualSelection(editor: IVimEditor): void {
    const selection = this.getVisualSelection(editor);
    if (!selection || selection.trim().length === 0) {
      editor.mode = EditorMode.Normal;
      return;
    }
    
    editor.searchKeyword = selection;
    const fastSearchHandler = editor.modeHandlerRegistry.getHandler(EditorMode.FastSearch);
    fastSearchHandler.findAllMatches(editor);
    
    if (editor.searchMatches.length === 0) {
      editor.mode = EditorMode.Normal;
      return;
    }
    
    editor.currentMatchIndex = 0;
    editor.mode = EditorMode.FastSearch;
    editor.cursorY = editor.searchMatches[0].y;
    editor.cursorX = editor.searchMatches[0].x;
    editor.searchHistory.push({
      keyword: editor.searchKeyword,
      matches: [...editor.searchMatches]
    });
  }
}

