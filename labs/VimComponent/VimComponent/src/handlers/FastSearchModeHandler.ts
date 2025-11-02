import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class FastSearchModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.FastSearch;
  
  shouldPreventDefault(key: string): boolean {
    return true;
  }
  
  private getKeyPatterns(editor: IVimEditor) {
    // Patterns ordered by priority
    return [
      // Cursor movement patterns (only within match boundaries)
      { pattern: /^(h|ArrowLeft)$/, action: () => { this.moveCursorLeftInMatch(editor); }, requireMatch: true },
      { pattern: /^(l|ArrowRight)$/, action: () => { this.moveCursorRightInMatch(editor); }, requireMatch: true },
      { pattern: /^(0|Home)$/, action: () => { this.moveCursorToMatchStart(editor); }, requireMatch: true },
      { pattern: /^(\$|End)$/, action: () => { this.moveCursorToMatchEnd(editor); }, requireMatch: true },
      { pattern: /^x$/, action: () => { this.searchModeDelete(editor); }, requireMatch: true },
      { pattern: /^d$/, action: () => { this.searchModeDeleteAll(editor); }, requireMatch: true },
      
      // General search mode commands
      { pattern: /^Escape$/, action: () => { 
        editor.mode = EditorMode.Normal;
        editor.searchKeyword = '';
        editor.searchMatches = [];
        editor.currentMatchIndex = -1;
      }, requireMatch: false },
      { pattern: /^n$/, action: () => { this.jumpToNextMatch(editor); }, requireMatch: false },
      { pattern: /^N$/, action: () => { this.jumpToPreviousMatch(editor); }, requireMatch: false },
      { pattern: /^b$/, action: () => { this.clearSearchMarks(editor); }, requireMatch: false },
      { pattern: /^u$/, action: () => { this.restoreSearchMarks(editor); }, requireMatch: false },
      { pattern: /^i$/, action: () => { 
        editor.saveHistory();
        editor.mode = EditorMode.MultiInsert;
      }, requireMatch: false },
      { pattern: /^a$/, action: () => { 
        if (editor.searchMatches.length > 0) {
          const currentMatch = editor.searchMatches[editor.currentMatchIndex];
          const matchEndX = currentMatch.x + editor.searchKeyword.length;
          if (editor.cursorX < matchEndX) {
            editor.cursorX++;
          }
        }
        editor.saveHistory();
        editor.mode = EditorMode.MultiInsert;
      }, requireMatch: false },
    ];
  }
  
  handleKey(key: string, editor: IVimEditor): void {
    const keyPatterns = this.getKeyPatterns(editor);
    const hasMatch = editor.currentMatchIndex >= 0 && editor.searchMatches.length > 0;
    
    // Try to match patterns in order
    for (const { pattern, action, requireMatch } of keyPatterns) {
      if (requireMatch && !hasMatch) {
        continue;
      }
      
      if (pattern.test(key)) {
        action();
        return;
      }
    }
  }
  
  private moveCursorLeftInMatch(editor: IVimEditor): void {
    const match = editor.searchMatches[editor.currentMatchIndex];
    if (editor.cursorX > match.x) {
      editor.cursorX--;
    }
  }
  
  private moveCursorRightInMatch(editor: IVimEditor): void {
    const match = editor.searchMatches[editor.currentMatchIndex];
    const matchEndX = match.x + editor.searchKeyword.length;
    if (editor.cursorX < matchEndX - 1) {
      editor.cursorX++;
    }
  }
  
  private moveCursorToMatchStart(editor: IVimEditor): void {
    const match = editor.searchMatches[editor.currentMatchIndex];
    editor.cursorX = match.x;
  }
  
  private moveCursorToMatchEnd(editor: IVimEditor): void {
    const match = editor.searchMatches[editor.currentMatchIndex];
    const matchEndX = match.x + editor.searchKeyword.length;
    editor.cursorX = matchEndX - 1;
  }
  
  findAllMatches(editor: IVimEditor): void {
    editor.searchMatches = [];
    const keyword = editor.searchKeyword;
    
    for (let y = 0; y < editor.content.length; y++) {
      const line = editor.content[y];
      let startIndex = 0;
      
      while (true) {
        const index = line.indexOf(keyword, startIndex);
        if (index === -1) break;
        
        editor.searchMatches.push({ y, x: index });
        startIndex = index + 1;
      }
    }
  }
  
  private jumpToNextMatch(editor: IVimEditor): void {
    if (editor.searchMatches.length === 0) return;
    
    editor.currentMatchIndex = (editor.currentMatchIndex + 1) % editor.searchMatches.length;
    const match = editor.searchMatches[editor.currentMatchIndex];
    editor.cursorY = match.y;
    editor.cursorX = match.x;
  }
  
  private jumpToPreviousMatch(editor: IVimEditor): void {
    if (editor.searchMatches.length === 0) return;
    
    editor.currentMatchIndex = (editor.currentMatchIndex - 1 + editor.searchMatches.length) % editor.searchMatches.length;
    const match = editor.searchMatches[editor.currentMatchIndex];
    editor.cursorY = match.y;
    editor.cursorX = match.x;
  }
  
  private clearSearchMarks(editor: IVimEditor): void {
    if (editor.currentMatchIndex < 0 || editor.searchMatches.length === 0) {
      return;
    }
    
    const removedMatch = editor.searchMatches[editor.currentMatchIndex];
    editor.searchHistory.push({
      keyword: editor.searchKeyword,
      matches: [removedMatch]
    });
    
    editor.searchMatches.splice(editor.currentMatchIndex, 1);
    
    if (editor.searchMatches.length === 0) {
      editor.searchKeyword = '';
      editor.currentMatchIndex = -1;
      editor.mode = EditorMode.Normal;
    } else {
      if (editor.currentMatchIndex >= editor.searchMatches.length) {
        editor.currentMatchIndex = 0;
      }
      const nextMatch = editor.searchMatches[editor.currentMatchIndex];
      editor.cursorY = nextMatch.y;
      editor.cursorX = nextMatch.x;
    }
  }
  
  private restoreSearchMarks(editor: IVimEditor): void {
    if (editor.searchHistory.length === 0) return;
    
    const lastSearch = editor.searchHistory.pop()!;
    
    for (const match of lastSearch.matches) {
      editor.searchMatches.push(match);
    }
    
    editor.searchMatches.sort((a: any, b: any) => {
      if (a.y !== b.y) return a.y - b.y;
      return a.x - b.x;
    });
    
    if (editor.searchKeyword.length === 0 && lastSearch.keyword) {
      editor.searchKeyword = lastSearch.keyword;
    }
    
    editor.currentMatchIndex = 0;
    editor.mode = EditorMode.FastSearch;
    
    if (editor.searchMatches.length > 0) {
      editor.cursorY = editor.searchMatches[0].y;
      editor.cursorX = editor.searchMatches[0].x;
    }
  }
  
  private searchModeDelete(editor: IVimEditor): void {
    if (editor.currentMatchIndex < 0 || editor.searchMatches.length === 0) {
      return;
    }
    
    editor.saveHistory();
    
    const match = editor.searchMatches[editor.currentMatchIndex];
    const matchEndX = match.x + editor.searchKeyword.length;
    const offsetInMatch = editor.cursorX - match.x;
    
    for (let i = editor.searchMatches.length - 1; i >= 0; i--) {
      const m = editor.searchMatches[i];
      const line = editor.content[m.y];
      const deletePos = m.x + offsetInMatch;
      
      if (deletePos < m.x + editor.searchKeyword.length) {
        editor.content[m.y] = line.substring(0, deletePos) + line.substring(deletePos + 1);
      }
    }
    
    editor.searchKeyword = editor.searchKeyword.substring(0, offsetInMatch) + 
                         editor.searchKeyword.substring(offsetInMatch + 1);
    
    if (editor.searchKeyword.length === 0) {
      editor.searchMatches = [];
      editor.currentMatchIndex = -1;
      editor.mode = EditorMode.Normal;
    } else if (editor.cursorX >= match.x + editor.searchKeyword.length) {
      editor.cursorX = match.x + editor.searchKeyword.length - 1;
    }
    
    editor.emitChange();
  }
  
  private searchModeDeleteAll(editor: IVimEditor): void {
    if (editor.currentMatchIndex < 0 || editor.searchMatches.length === 0) {
      return;
    }
    
    editor.saveHistory();
    
    for (let i = editor.searchMatches.length - 1; i >= 0; i--) {
      const match = editor.searchMatches[i];
      const line = editor.content[match.y];
      
      editor.content[match.y] = line.substring(0, match.x) + 
                              line.substring(match.x + editor.searchKeyword.length);
    }
    
    editor.searchKeyword = '';
    editor.searchMatches = [];
    editor.currentMatchIndex = -1;
    editor.mode = EditorMode.Normal;
    editor.emitChange();
  }
}

