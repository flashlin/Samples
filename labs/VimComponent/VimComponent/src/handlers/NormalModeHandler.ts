import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class NormalModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Normal;
  
  shouldPreventDefault(key: string): boolean {
    return true;
  }
  
  private getCommandPatterns(editor: IVimEditor) {
    return [
      { pattern: 'gg', action: () => { editor.moveToFirstLine(); } },
      { pattern: 'diw', action: () => { editor.saveHistory(); this.deleteInnerWord(editor); } },
      { pattern: 'di%', action: () => { editor.saveHistory(); this.deleteInnerBracket(editor); } },
      { pattern: 'di`', action: () => { editor.saveHistory(); this.deleteInnerQuote(editor, '`'); } },
      { pattern: "di'", action: () => { editor.saveHistory(); this.deleteInnerQuote(editor, "'"); } },
      { pattern: 'di"', action: () => { editor.saveHistory(); this.deleteInnerQuote(editor, '"'); } },
      { pattern: 'da(', action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '(', ')'); } },
      { pattern: 'da)', action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '(', ')'); } },
      { pattern: 'da[', action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '[', ']'); } },
      { pattern: 'da]', action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '[', ']'); } },
      { pattern: 'da{', action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '{', '}'); } },
      { pattern: 'da}', action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '{', '}'); } },
      { pattern: 'da<', action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '<', '>'); } },
      { pattern: 'da>', action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '<', '>'); } },
      { pattern: 'da`', action: () => { editor.saveHistory(); this.deleteAroundQuote(editor, '`'); } },
      { pattern: "da'", action: () => { editor.saveHistory(); this.deleteAroundQuote(editor, "'"); } },
      { pattern: 'da"', action: () => { editor.saveHistory(); this.deleteAroundQuote(editor, '"'); } },
      { pattern: 'da%', action: () => { editor.saveHistory(); this.deleteAroundAnyBracket(editor); } },
      { pattern: 'dw', action: () => { editor.saveHistory(); editor.deleteWord(); } },
      { pattern: 'de', action: () => { editor.saveHistory(); editor.deleteToWordEnd(); } },
      { pattern: 'i', action: () => { editor.enterInsertMode(); } },
      { pattern: 't', action: () => { this.addTMark(editor); } },
      { pattern: 'T', action: () => { this.addTMarkNext(editor); } },
      { pattern: 'Escape', action: () => { this.clearTMarks(editor); } },
      { pattern: 'a', action: () => { 
        const currentLine = editor.content[editor.cursorY] || '';
        if (editor.cursorX < currentLine.length) {
          editor.cursorX += 1;
        }
        editor.mode = EditorMode.Insert;
        editor.updateInputPosition();
        editor.hiddenInput?.focus();
      } },
      { pattern: 'o', action: () => { editor.insertLineBelow(); editor.hiddenInput?.focus(); } },
      { pattern: 'p', action: () => { editor.saveHistory(); editor.pasteAfterCursor(); } },
      { pattern: 'v', action: () => { 
        editor.mode = EditorMode.Visual;
        editor.visualStartX = editor.cursorX;
        editor.visualStartY = editor.cursorY;
      } },
      { pattern: 'V', action: () => { 
        editor.mode = EditorMode.VisualLine;
        editor.visualStartX = editor.cursorX;
        editor.visualStartY = editor.cursorY;
      } },
      { pattern: 'f', action: () => { 
        editor.previousMode = EditorMode.Normal;
        editor.mode = EditorMode.FastJump;
        editor.fastJumpMatches = [];
        editor.fastJumpInput = '';
      } },
      { pattern: 'u', action: () => { editor.undo(); } },
      { pattern: '%', action: () => { editor.jumpToMatchingBracket(); } },
    ];
  }
  
  handleKey(key: string, editor: IVimEditor): void {
    if (editor.keyBuffer === '' && editor.handleMovement(key)) {
      return;
    }
    
    editor.keyBuffer += key;
    this.processKeyBuffer(editor);
  }
  
  private processKeyBuffer(editor: IVimEditor): boolean {
    const commandPatterns = this.getCommandPatterns(editor);
    const sortedPatterns = [...commandPatterns].sort((a, b) => b.pattern.length - a.pattern.length);
    
    for (const { pattern, action } of sortedPatterns) {
      if (editor.keyBuffer === pattern) {
        editor.keyBuffer = '';
        action();
        return true;
      }
    }
    
    const dNumberJMatch = /^d(\d+)j$/.exec(editor.keyBuffer);
    if (dNumberJMatch) {
      const count = parseInt(dNumberJMatch[1], 10);
      editor.keyBuffer = '';
      editor.saveHistory();
      editor.deleteLinesDown(count);
      return true;
    }
    
    const dNumberKMatch = /^d(\d+)k$/.exec(editor.keyBuffer);
    if (dNumberKMatch) {
      const count = parseInt(dNumberKMatch[1], 10);
      editor.keyBuffer = '';
      editor.saveHistory();
      editor.deleteLinesUp(count);
      return true;
    }
    
    const numberGMatch = /^(\d+)G$/.exec(editor.keyBuffer);
    if (numberGMatch) {
      const lineNumber = parseInt(numberGMatch[1], 10) - 1;
      editor.keyBuffer = '';
      editor['moveToLine'](lineNumber);
      return true;
    }
    
    const ddMatch = /^d{2,}$/.exec(editor.keyBuffer);
    if (ddMatch) {
      if (editor.keyBuffer === 'dd') {
        editor.keyBuffer = '';
        editor.saveHistory();
        editor['deleteLine']();
        return true;
      }
    }
    
    if (editor.keyBuffer.length > 10) {
      editor.keyBuffer = '';
      return false;
    }
    
    return false;
  }
  
  addTMark(editor: IVimEditor): void {
    const existingIndex = editor.tMarks.findIndex(
      (mark: any) => mark.y === editor.cursorY && mark.x === editor.cursorX
    );
    
    if (existingIndex === -1) {
      editor.tMarks.push({ y: editor.cursorY, x: editor.cursorX });
      editor.tMarks.sort((a: any, b: any) => {
        if (a.y !== b.y) return a.y - b.y;
        return a.x - b.x;
      });
    }
  }
  
  addTMarkNext(editor: IVimEditor): void {
    const currentLine = editor.content[editor.cursorY] || '';
    // Calculate next position
    let nextX = editor.cursorX + 1;
    let nextY = editor.cursorY;
    
    // If next position exceeds current line length, move to next line
    if (nextX >= currentLine.length) {
      if (editor.cursorY < editor.content.length - 1) {
        nextY = editor.cursorY + 1;
        nextX = 0;
      } else {
        // At the end of the last line, stay at current position
        nextX = editor.cursorX;
      }
    }
    
    // Check if mark already exists at this position
    const existingIndex = editor.tMarks.findIndex(
      (mark: any) => mark.y === nextY && mark.x === nextX
    );
    
    if (existingIndex === -1) {
      editor.tMarks.push({ y: nextY, x: nextX });
      editor.tMarks.sort((a: any, b: any) => {
        if (a.y !== b.y) return a.y - b.y;
        return a.x - b.x;
      });
    }
  }
  
  clearTMarks(editor: IVimEditor): void {
    editor.tMarks = [];
  }
  
  private deleteInnerWord(editor: IVimEditor): void {
    const range = editor.getInnerWordRange();
    if (!range) {
      return;
    }
    
    const currentLine = editor.content[range.y];
    const beforeWord = currentLine.substring(0, range.startX);
    const afterWord = currentLine.substring(range.endX + 1);
    
    editor.content[range.y] = beforeWord + afterWord;
    
    editor.cursorX = range.startX;
    if (editor.cursorX >= editor.content[range.y].length && editor.content[range.y].length > 0) {
      editor.cursorX = editor.content[range.y].length - 1;
    }
    if (editor.cursorX < 0) {
      editor.cursorX = 0;
    }
  }
  
  private deleteInnerQuote(editor: IVimEditor, quoteChar: string): void {
    const range = editor.getInnerQuoteRange(quoteChar);
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = editor.content[range.startY];
      const beforeQuote = currentLine.substring(0, range.startX + 1);
      const afterQuote = currentLine.substring(range.endX);
      
      editor.content[range.startY] = beforeQuote + afterQuote;
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX + 1;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    } else {
      const firstLine = editor.content[range.startY];
      const lastLine = editor.content[range.endY];
      
      const beforeQuote = firstLine.substring(0, range.startX + 1);
      const afterQuote = lastLine.substring(range.endX);
      
      editor.content[range.startY] = beforeQuote + afterQuote;
      editor.content.splice(range.startY + 1, range.endY - range.startY);
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX + 1;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    }
  }
  
  private deleteInnerBracket(editor: IVimEditor): void {
    const range = editor.getInnerBracketRange();
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = editor.content[range.startY];
      const beforeBracket = currentLine.substring(0, range.startX + 1);
      const afterBracket = currentLine.substring(range.endX);
      
      editor.content[range.startY] = beforeBracket + afterBracket;
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX + 1;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    } else {
      const firstLine = editor.content[range.startY];
      const lastLine = editor.content[range.endY];
      
      const beforeBracket = firstLine.substring(0, range.startX + 1);
      const afterBracket = lastLine.substring(range.endX);
      
      editor.content[range.startY] = beforeBracket + afterBracket;
      editor.content.splice(range.startY + 1, range.endY - range.startY);
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX + 1;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    }
  }
  
  private deleteAroundBracket(editor: IVimEditor, openChar: string, closeChar: string): void {
    const range = editor.findInnerBracketRange(openChar, closeChar);
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = editor.content[range.startY];
      const beforeBracket = currentLine.substring(0, range.startX);
      const afterBracket = currentLine.substring(range.endX + 1);
      
      editor.content[range.startY] = beforeBracket + afterBracket;
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    } else {
      const firstLine = editor.content[range.startY];
      const lastLine = editor.content[range.endY];
      
      const beforeBracket = firstLine.substring(0, range.startX);
      const afterBracket = lastLine.substring(range.endX + 1);
      
      editor.content[range.startY] = beforeBracket + afterBracket;
      editor.content.splice(range.startY + 1, range.endY - range.startY);
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    }
  }
  
  private deleteAroundQuote(editor: IVimEditor, quoteChar: string): void {
    const range = editor.getInnerQuoteRange(quoteChar);
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = editor.content[range.startY];
      const beforeQuote = currentLine.substring(0, range.startX);
      const afterQuote = currentLine.substring(range.endX + 1);
      
      editor.content[range.startY] = beforeQuote + afterQuote;
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    } else {
      const firstLine = editor.content[range.startY];
      const lastLine = editor.content[range.endY];
      
      const beforeQuote = firstLine.substring(0, range.startX);
      const afterQuote = lastLine.substring(range.endX + 1);
      
      editor.content[range.startY] = beforeQuote + afterQuote;
      editor.content.splice(range.startY + 1, range.endY - range.startY);
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    }
  }
  
  private deleteAroundAnyBracket(editor: IVimEditor): void {
    const range = editor.getInnerBracketRange();
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = editor.content[range.startY];
      const beforeBracket = currentLine.substring(0, range.startX);
      const afterBracket = currentLine.substring(range.endX + 1);
      
      editor.content[range.startY] = beforeBracket + afterBracket;
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    } else {
      const firstLine = editor.content[range.startY];
      const lastLine = editor.content[range.endY];
      
      const beforeBracket = firstLine.substring(0, range.startX);
      const afterBracket = lastLine.substring(range.endX + 1);
      
      editor.content[range.startY] = beforeBracket + afterBracket;
      editor.content.splice(range.startY + 1, range.endY - range.startY);
      
      editor.cursorY = range.startY;
      editor.cursorX = range.startX;
      if (editor.cursorX >= editor.content[editor.cursorY].length && editor.content[editor.cursorY].length > 0) {
        editor.cursorX = editor.content[editor.cursorY].length - 1;
      }
    }
  }
}

