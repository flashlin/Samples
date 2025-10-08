import { EditorMode, BaseModeHandler } from '../vimEditorTypes';

export class NormalModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Normal;
  
  private getCommandPatterns(editor: any) {
    return [
      { pattern: 'gg', action: () => { editor['moveToFirstLine'](); } },
      { pattern: 'diw', action: () => { editor['saveHistory'](); editor['deleteInnerWord'](); } },
      { pattern: 'di%', action: () => { editor['saveHistory'](); editor['deleteInnerBracket'](); } },
      { pattern: 'di`', action: () => { editor['saveHistory'](); editor['deleteInnerQuote']('`'); } },
      { pattern: "di'", action: () => { editor['saveHistory'](); editor['deleteInnerQuote']("'"); } },
      { pattern: 'di"', action: () => { editor['saveHistory'](); editor['deleteInnerQuote']('"'); } },
      { pattern: 'da(', action: () => { editor['saveHistory'](); editor['deleteAroundBracket']('(', ')'); } },
      { pattern: 'da)', action: () => { editor['saveHistory'](); editor['deleteAroundBracket']('(', ')'); } },
      { pattern: 'da[', action: () => { editor['saveHistory'](); editor['deleteAroundBracket']('[', ']'); } },
      { pattern: 'da]', action: () => { editor['saveHistory'](); editor['deleteAroundBracket']('[', ']'); } },
      { pattern: 'da{', action: () => { editor['saveHistory'](); editor['deleteAroundBracket']('{', '}'); } },
      { pattern: 'da}', action: () => { editor['saveHistory'](); editor['deleteAroundBracket']('{', '}'); } },
      { pattern: 'da<', action: () => { editor['saveHistory'](); editor['deleteAroundBracket']('<', '>'); } },
      { pattern: 'da>', action: () => { editor['saveHistory'](); editor['deleteAroundBracket']('<', '>'); } },
      { pattern: 'da`', action: () => { editor['saveHistory'](); editor['deleteAroundQuote']('`'); } },
      { pattern: "da'", action: () => { editor['saveHistory'](); editor['deleteAroundQuote']("'"); } },
      { pattern: 'da"', action: () => { editor['saveHistory'](); editor['deleteAroundQuote']('"'); } },
      { pattern: 'da%', action: () => { editor['saveHistory'](); editor['deleteAroundAnyBracket'](); } },
      { pattern: 'dw', action: () => { editor['saveHistory'](); editor['deleteWord'](); } },
      { pattern: 'de', action: () => { editor['saveHistory'](); editor['deleteToWordEnd'](); } },
      { pattern: 'i', action: () => { editor['enterInsertMode'](); } },
      { pattern: 't', action: () => { this.addTMark(editor); } },
      { pattern: 'T', action: () => { this.clearTMarks(editor); } },
      { pattern: 'a', action: () => { 
        const currentLine = editor.content[editor.cursorY] || '';
        if (editor.cursorX < currentLine.length) {
          editor.cursorX += 1;
        }
        editor.mode = EditorMode.Insert;
        editor['updateInputPosition']();
        editor['hiddenInput']?.focus();
      } },
      { pattern: 'o', action: () => { editor['insertLineBelow'](); editor['hiddenInput']?.focus(); } },
      { pattern: 'p', action: () => { editor['saveHistory'](); editor['pasteAfterCursor'](); } },
      { pattern: 'v', action: () => { 
        editor.mode = EditorMode.Visual;
        editor['visualStartX'] = editor.cursorX;
        editor['visualStartY'] = editor.cursorY;
      } },
      { pattern: 'V', action: () => { 
        editor.mode = EditorMode.VisualLine;
        editor['visualStartX'] = editor.cursorX;
        editor['visualStartY'] = editor.cursorY;
      } },
      { pattern: 'f', action: () => { 
        editor['previousMode'] = EditorMode.Normal;
        editor.mode = EditorMode.FastJump;
        editor['fastJumpMatches'] = [];
        editor['fastJumpInput'] = '';
      } },
      { pattern: 'u', action: () => { editor['undo'](); } },
      { pattern: '%', action: () => { editor['jumpToMatchingBracket'](); } },
    ];
  }
  
  handleKey(key: string, editor: any): void {
    if (editor['keyBuffer'] === '' && editor['handleMovement'](key)) {
      return;
    }
    
    editor['keyBuffer'] += key;
    this.processKeyBuffer(editor);
  }
  
  private processKeyBuffer(editor: any): boolean {
    const commandPatterns = this.getCommandPatterns(editor);
    const sortedPatterns = [...commandPatterns].sort((a, b) => b.pattern.length - a.pattern.length);
    
    for (const { pattern, action } of sortedPatterns) {
      if (editor['keyBuffer'] === pattern) {
        editor['keyBuffer'] = '';
        action();
        return true;
      }
    }
    
    const dNumberJMatch = /^d(\d+)j$/.exec(editor['keyBuffer']);
    if (dNumberJMatch) {
      const count = parseInt(dNumberJMatch[1], 10);
      editor['keyBuffer'] = '';
      editor['saveHistory']();
      editor['deleteLinesDown'](count);
      return true;
    }
    
    const dNumberKMatch = /^d(\d+)k$/.exec(editor['keyBuffer']);
    if (dNumberKMatch) {
      const count = parseInt(dNumberKMatch[1], 10);
      editor['keyBuffer'] = '';
      editor['saveHistory']();
      editor['deleteLinesUp'](count);
      return true;
    }
    
    const numberGMatch = /^(\d+)G$/.exec(editor['keyBuffer']);
    if (numberGMatch) {
      const lineNumber = parseInt(numberGMatch[1], 10) - 1;
      editor['keyBuffer'] = '';
      editor['moveToLine'](lineNumber);
      return true;
    }
    
    const ddMatch = /^d{2,}$/.exec(editor['keyBuffer']);
    if (ddMatch) {
      if (editor['keyBuffer'] === 'dd') {
        editor['keyBuffer'] = '';
        editor['saveHistory']();
        editor['deleteLine']();
        return true;
      }
    }
    
    if (editor['keyBuffer'].length > 10) {
      editor['keyBuffer'] = '';
      return false;
    }
    
    return false;
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

