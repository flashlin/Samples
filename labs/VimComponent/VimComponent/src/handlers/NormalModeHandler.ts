import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class NormalModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Normal;
  
  shouldPreventDefault(key: string): boolean {
    return true;
  }
  
  private getCommandPatterns(editor: IVimEditor) {
    // Patterns ordered by priority (most specific first)
    // Using regex patterns for more flexible matching
    return [
      // Multi-character delete commands with text objects (highest priority)
      { pattern: /^diw$/, action: () => { editor.saveHistory(); this.deleteInnerWord(editor); } },
      { pattern: /^di%$/, action: () => { editor.saveHistory(); this.deleteInnerBracket(editor); } },
      { pattern: /^di`$/, action: () => { editor.saveHistory(); this.deleteInnerQuote(editor, '`'); } },
      { pattern: /^di'$/, action: () => { editor.saveHistory(); this.deleteInnerQuote(editor, "'"); } },
      { pattern: /^di"$/, action: () => { editor.saveHistory(); this.deleteInnerQuote(editor, '"'); } },
      { pattern: /^da\($/, action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '(', ')'); } },
      { pattern: /^da\)$/, action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '(', ')'); } },
      { pattern: /^da\[$/, action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '[', ']'); } },
      { pattern: /^da\]$/, action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '[', ']'); } },
      { pattern: /^da\{$/, action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '{', '}'); } },
      { pattern: /^da\}$/, action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '{', '}'); } },
      { pattern: /^da<$/, action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '<', '>'); } },
      { pattern: /^da>$/, action: () => { editor.saveHistory(); this.deleteAroundBracket(editor, '<', '>'); } },
      { pattern: /^da`$/, action: () => { editor.saveHistory(); this.deleteAroundQuote(editor, '`'); } },
      { pattern: /^da'$/, action: () => { editor.saveHistory(); this.deleteAroundQuote(editor, "'"); } },
      { pattern: /^da"$/, action: () => { editor.saveHistory(); this.deleteAroundQuote(editor, '"'); } },
      { pattern: /^da%$/, action: () => { editor.saveHistory(); this.deleteAroundAnyBracket(editor); } },
      
      // Delete with count and direction
      { pattern: /^d(\d+)j$/, action: (match: RegExpMatchArray) => { 
        const count = parseInt(match[1], 10);
        editor.saveHistory();
        editor.deleteLinesDown(count);
      } },
      { pattern: /^d(\d+)k$/, action: (match: RegExpMatchArray) => { 
        const count = parseInt(match[1], 10);
        editor.saveHistory();
        editor.deleteLinesUp(count);
      } },
      
      // Double character commands
      { pattern: /^dd$/, action: () => { 
        editor.saveHistory();
        editor['deleteLine']();
      } },
      { pattern: /^gg$/, action: () => { editor.moveToFirstLine(); } },
      
      // Two-character delete commands
      { pattern: /^dw$/, action: () => { editor.saveHistory(); editor.deleteWord(); } },
      { pattern: /^de$/, action: () => { editor.saveHistory(); editor.deleteToWordEnd(); } },
      
      // Go to line number
      { pattern: /^(\d+)G$/, action: (match: RegExpMatchArray) => { 
        const lineNumber = parseInt(match[1], 10) - 1;
        editor['moveToLine'](lineNumber);
      } },
      
      // Single character commands (lowest priority)
      { pattern: /^Escape$/, action: () => { this.clearTMarks(editor); } },
      { pattern: /^i$/, action: () => { editor.enterInsertMode(); } },
      { pattern: /^t$/, action: () => { this.addTMark(editor); } },
      { pattern: /^T$/, action: () => { this.addTMarkNext(editor); } },
      { pattern: /^a$/, action: () => { 
        const currentLine = editor.content[editor.cursorY] || '';
        if (editor.cursorX < currentLine.length) {
          editor.cursorX += 1;
        }
        editor.mode = EditorMode.Insert;
        editor.updateInputPosition();
        editor.hiddenInput?.focus();
      } },
      { pattern: /^o$/, action: () => { editor.insertLineBelow(); editor.hiddenInput?.focus(); } },
      { pattern: /^p$/, action: () => { 
        editor.saveHistory(); 
        if (editor.multiCursorClipboard.length > 0 && editor.tMarks.length > 0) {
          this.pasteMultiCursor(editor);
          // After pasting, enter TInsert mode
          editor.mode = EditorMode.TInsert;
          const lastMark = editor.tMarks[editor.tMarks.length - 1];
          editor.cursorY = lastMark.y;
          editor.cursorX = lastMark.x;
          editor.updateInputPosition();
          editor.hiddenInput?.focus();
        } else {
          editor.pasteAfterCursor(); 
        }
      } },
      { pattern: /^v$/, action: () => { 
        // Check if current cursor position is in tMarks
        const isInTMark = editor.tMarks.some(
          (mark: any) => mark.x === editor.cursorX && mark.y === editor.cursorY
        );
        
        if (isInTMark && editor.tMarks.length > 0) {
          // Enter TVisual mode (multi-cursor visual based on tMarks)
          editor.mode = EditorMode.TVisual;
          editor.visualStartX = editor.cursorX;
          editor.visualStartY = editor.cursorY;
        } else {
          // Normal visual mode
          editor.mode = EditorMode.Visual;
          editor.visualStartX = editor.cursorX;
          editor.visualStartY = editor.cursorY;
        }
      } },
      { pattern: /^V$/, action: () => { 
        editor.mode = EditorMode.VisualLine;
        editor.visualStartX = editor.cursorX;
        editor.visualStartY = editor.cursorY;
      } },
      { pattern: /^f$/, action: () => { 
        editor.previousMode = EditorMode.Normal;
        editor.mode = EditorMode.FastJump;
        editor.fastJumpMatches = [];
        editor.fastJumpInput = '';
      } },
      { pattern: /^:$/, action: () => { editor.mode = EditorMode.Command; } },
      { pattern: /^u$/, action: () => { editor.undo(); } },
      { pattern: /^%$/, action: () => { editor.jumpToMatchingBracket(); } },
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
    
    // Try to match patterns in order (already ordered by priority)
    for (const { pattern, action } of commandPatterns) {
      const match = pattern.exec(editor.keyBuffer);
      if (match) {
        editor.keyBuffer = '';
        // Pass match array to action if it accepts it (for capture groups)
        action(match);
        return true;
      }
    }
    
    // If buffer gets too long without matching, clear it
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
  
  pasteMultiCursor(editor: IVimEditor): void {
    // Paste each clipboard item at each tMark position
    const sortedMarks = [...editor.tMarks].sort((a: any, b: any) => {
      if (a.y !== b.y) return b.y - a.y; // Start from bottom to avoid position shifts
      return b.x - a.x;
    });
    
    for (let i = 0; i < sortedMarks.length && i < editor.multiCursorClipboard.length; i++) {
      const mark = sortedMarks[i];
      const text = editor.multiCursorClipboard[i];
      
      if (!text) continue;
      
      const currentLine = editor.content[mark.y] || '';
      const lines = text.split('\n');
      
      if (lines.length === 1) {
        // Single line paste
        let insertPosition = Math.min(mark.x + 1, currentLine.length);
        if (currentLine.length === 0) {
          insertPosition = 0;
        }
        
        const beforeInsert = currentLine.substring(0, insertPosition);
        const afterInsert = currentLine.substring(insertPosition);
        editor.content[mark.y] = beforeInsert + text + afterInsert;
      } else {
        // Multi-line paste
        let insertPosition = Math.min(mark.x + 1, currentLine.length);
        if (currentLine.length === 0) {
          insertPosition = 0;
        }
        
        const beforeInsert = currentLine.substring(0, insertPosition);
        const afterInsert = currentLine.substring(insertPosition);
        
        editor.content[mark.y] = beforeInsert + lines[0];
        
        for (let j = 1; j < lines.length - 1; j++) {
          editor.content.splice(mark.y + j, 0, lines[j]);
        }
        
        if (lines.length > 1) {
          const lastLine = lines[lines.length - 1];
          editor.content.splice(mark.y + lines.length - 1, 0, lastLine + afterInsert);
        }
      }
    }
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
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

