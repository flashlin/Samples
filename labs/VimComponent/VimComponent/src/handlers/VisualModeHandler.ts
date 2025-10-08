import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class VisualModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Visual;
  
  handleKey(key: string, editor: IVimEditor): void {
    if (editor.visualKeyBuffer === '' && editor.handleMovement(key)) {
      return;
    }
    
    if (key === 'Escape') {
      editor.mode = EditorMode.Normal;
      editor.visualKeyBuffer = '';
      return;
    }
    
    if (key === 'i' && editor.visualKeyBuffer === '') {
      editor.visualKeyBuffer = 'i';
      return;
    }
    
    if (editor.visualKeyBuffer === 'i' && (key === '`' || key === "'" || key === '"')) {
      this.selectInnerQuote(editor, key);
      editor.visualKeyBuffer = '';
      return;
    }
    
    if (editor.visualKeyBuffer === 'i' && key === 'w') {
      this.selectInnerWord(editor);
      editor.visualKeyBuffer = '';
      return;
    }
    
    editor.visualKeyBuffer = '';
    
    switch (key) {
      case 'y':
        this.yankVisualSelection(editor);
        editor.mode = EditorMode.Normal;
        break;
      case 'c':
      case 'd':
      case 'x':
        this.cutVisualSelection(editor);
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
  
  private yankVisualSelection(editor: IVimEditor): void {
    const selection = this.getVisualSelection(editor);
    navigator.clipboard.writeText(selection);
  }
  
  private cutVisualSelection(editor: IVimEditor): void {
    const selection = this.getVisualSelection(editor);
    navigator.clipboard.writeText(selection);
    
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

