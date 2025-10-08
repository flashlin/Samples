import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class FastSearchModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.FastSearch;
  
  handleKey(key: string, editor: IVimEditor): void {
    if (editor.currentMatchIndex >= 0 && editor.searchMatches.length > 0) {
      const match = editor.searchMatches[editor.currentMatchIndex];
      const matchEndX = match.x + editor.searchKeyword.length;
      
      if (key === 'h' || key === 'ArrowLeft') {
        if (editor.cursorX > match.x) {
          editor.cursorX--;
        }
        return;
      }
      
      if (key === 'l' || key === 'ArrowRight') {
        if (editor.cursorX < matchEndX - 1) {
          editor.cursorX++;
        }
        return;
      }
      
      if (key === '0' || key === 'Home') {
        editor.cursorX = match.x;
        return;
      }
      
      if (key === '$' || key === 'End') {
        editor.cursorX = matchEndX - 1;
        return;
      }
      
      if (key === 'x') {
        this.searchModeDelete(editor);
        return;
      }
      
      if (key === 'd') {
        this.searchModeDeleteAll(editor);
        return;
      }
    }
    
    switch (key) {
      case 'Escape':
        editor.mode = EditorMode.Normal;
        editor.searchKeyword = '';
        editor.searchMatches = [];
        editor.currentMatchIndex = -1;
        break;
      case 'n':
        this.jumpToNextMatch(editor);
        break;
      case 'N':
        this.jumpToPreviousMatch(editor);
        break;
      case 'b':
        this.clearSearchMarks(editor);
        break;
      case 'u':
        this.restoreSearchMarks(editor);
        break;
      case 'i':
        editor.saveHistory();
        editor.mode = EditorMode.MultiInsert;
        break;
      case 'a':
        if (editor.searchMatches.length > 0) {
          const currentMatch = editor.searchMatches[editor.currentMatchIndex];
          const matchEndX = currentMatch.x + editor.searchKeyword.length;
          if (editor.cursorX < matchEndX) {
            editor.cursorX++;
          }
        }
        editor.saveHistory();
        editor.mode = EditorMode.MultiInsert;
        break;
    }
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
  }
}

