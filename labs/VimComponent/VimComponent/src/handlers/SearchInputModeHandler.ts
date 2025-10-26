import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class SearchInputModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.SearchInput;
  
  onEnter(editor: IVimEditor): void {
    editor.searchInput = '/';
    if (editor.hiddenInput) {
      const p5Instance = editor.p5Instance;
      if (p5Instance) {
        const statusBarHeight = 24;
        const editorHeight = p5Instance.height - statusBarHeight;
        const statusY = editorHeight;
        
        editor.hiddenInput.style.left = '10px';
        editor.hiddenInput.style.top = `${statusY + 3}px`;
      }
      
      editor.hiddenInput.focus();
    }
  }
  
  onExit(editor: IVimEditor): void {
    const host = (editor as any).shadowRoot?.host as HTMLElement;
    if (host) {
      host.focus();
    }
  }
  
  shouldPreventDefault(key: string): boolean {
    if (key === 'Enter' || key === 'Escape' || key === 'Backspace') {
      return true;
    }
    return false;
  }
  
  private getKeyPatterns(editor: IVimEditor) {
    return [
      { 
        pattern: /^Enter$/, 
        action: () => { 
          const searchKeyword = editor.searchInput.substring(1);
          if (searchKeyword.length > 0) {
            editor.searchKeyword = searchKeyword;
            this.performSearch(editor);
            if (editor.searchMatches.length > 0) {
              editor.currentMatchIndex = 0;
              const firstMatch = editor.searchMatches[0];
              editor.cursorY = firstMatch.y;
              editor.cursorX = firstMatch.x;
            }
          }
          editor.searchInput = '';
          editor.mode = EditorMode.Normal;
        } 
      },
      { 
        pattern: /^Escape$/, 
        action: () => { 
          editor.searchInput = '';
          editor.searchKeyword = '';
          editor.searchMatches = [];
          editor.currentMatchIndex = -1;
          editor.mode = EditorMode.Normal;
        } 
      },
      { 
        pattern: /^Backspace$/, 
        action: () => { 
          if (editor.searchInput.length > 1) {
            editor.searchInput = editor.searchInput.slice(0, -1);
            const searchKeyword = editor.searchInput.substring(1);
            if (searchKeyword.length > 0) {
              editor.searchKeyword = searchKeyword;
              this.performSearch(editor);
            } else {
              editor.searchKeyword = '';
              editor.searchMatches = [];
              editor.currentMatchIndex = -1;
            }
            this.updateInputPosition(editor);
          }
        } 
      },
    ];
  }
  
  handleKey(key: string, editor: IVimEditor): void {
    const keyPatterns = this.getKeyPatterns(editor);
    
    for (const { pattern, action } of keyPatterns) {
      if (pattern.test(key)) {
        action();
        if (editor.p5Instance) {
          editor.p5Instance.redraw();
        }
        return;
      }
    }
  }
  
  handleInput(editor: IVimEditor, value: string): void {
    editor.searchInput += value;
    const searchKeyword = editor.searchInput.substring(1);
    editor.searchKeyword = searchKeyword;
    
    if (searchKeyword.length > 0) {
      this.performSearch(editor);
    } else {
      editor.searchMatches = [];
      editor.currentMatchIndex = -1;
    }
    
    this.updateInputPosition(editor);
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  handleCompositionEnd(editor: IVimEditor, data: string): void {
    if (data) {
      editor.searchInput += data;
      const searchKeyword = editor.searchInput.substring(1);
      editor.searchKeyword = searchKeyword;
      
      if (searchKeyword.length > 0) {
        this.performSearch(editor);
      }
      
      this.updateInputPosition(editor);
      if (editor.p5Instance) {
        editor.p5Instance.redraw();
      }
    }
  }
  
  private performSearch(editor: IVimEditor): void {
    editor.searchMatches = [];
    const keyword = editor.searchKeyword;
    
    if (keyword.length === 0) {
      return;
    }
    
    for (let y = 0; y < editor.content.length; y++) {
      const line = editor.content[y];
      let startIndex = 0;
      
      while (true) {
        const index = line.toLowerCase().indexOf(keyword.toLowerCase(), startIndex);
        if (index === -1) break;
        
        editor.searchMatches.push({ y, x: index });
        startIndex = index + 1;
      }
    }
  }
  
  private updateInputPosition(editor: IVimEditor): void {
    if (editor.hiddenInput && editor.p5Instance) {
      const p5Instance = editor.p5Instance;
      const textWidth = p5Instance.textWidth(editor.searchInput);
      const statusBarHeight = 24;
      const editorHeight = p5Instance.height - statusBarHeight;
      const statusY = editorHeight;
      
      editor.hiddenInput.style.left = `${10 + textWidth}px`;
      editor.hiddenInput.style.top = `${statusY + 3}px`;
    }
  }
}

