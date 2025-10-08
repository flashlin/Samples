import { EditorMode, BaseModeHandler } from '../vimEditorTypes';

export class FastSearchModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.FastSearch;
  
  handleKey(key: string, editor: any): void {
    if (editor['currentMatchIndex'] >= 0 && editor['searchMatches'].length > 0) {
      const match = editor['searchMatches'][editor['currentMatchIndex']];
      const matchEndX = match.x + editor['searchKeyword'].length;
      
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
        editor['searchModeDelete']();
        return;
      }
      
      if (key === 'd') {
        editor['searchModeDeleteAll']();
        return;
      }
    }
    
    switch (key) {
      case 'Escape':
        editor.mode = EditorMode.Normal;
        editor['searchKeyword'] = '';
        editor['searchMatches'] = [];
        editor['currentMatchIndex'] = -1;
        break;
      case 'n':
        editor['jumpToNextMatch']();
        break;
      case 'N':
        editor['jumpToPreviousMatch']();
        break;
      case 'b':
        editor['clearSearchMarks']();
        break;
      case 'u':
        editor['restoreSearchMarks']();
        break;
      case 'i':
        editor['saveHistory']();
        editor.mode = EditorMode.MultiInsert;
        break;
      case 'a':
        if (editor['searchMatches'].length > 0) {
          const currentMatch = editor['searchMatches'][editor['currentMatchIndex']];
          const matchEndX = currentMatch.x + editor['searchKeyword'].length;
          if (editor.cursorX < matchEndX) {
            editor.cursorX++;
          }
        }
        editor['saveHistory']();
        editor.mode = EditorMode.MultiInsert;
        break;
    }
  }
}

