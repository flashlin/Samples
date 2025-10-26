import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class CommandModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Command;
  
  onEnter(editor: IVimEditor): void {
    editor.commandInput = ':';
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
          const command = editor.commandInput.substring(1);
          
          if (command === 'noh' || command === 'nohl' || command === 'nohlsearch') {
            editor.searchKeyword = '';
            editor.searchMatches = [];
            editor.currentMatchIndex = -1;
          } else {
            this.dispatchCommandEvent(editor, command);
          }
          
          editor.commandInput = '';
          editor.mode = EditorMode.Normal;
        } 
      },
      { 
        pattern: /^Escape$/, 
        action: () => { 
          editor.commandInput = '';
          editor.mode = EditorMode.Normal;
        } 
      },
      { 
        pattern: /^Backspace$/, 
        action: () => { 
          if (editor.commandInput.length > 1) {
            editor.commandInput = editor.commandInput.slice(0, -1);
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
    editor.commandInput += value;
    this.updateInputPosition(editor);
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  handleCompositionEnd(editor: IVimEditor, data: string): void {
    if (data) {
      editor.commandInput += data;
      this.updateInputPosition(editor);
      if (editor.p5Instance) {
        editor.p5Instance.redraw();
      }
    }
  }
  
  private updateInputPosition(editor: IVimEditor): void {
    if (editor.hiddenInput && editor.p5Instance) {
      const p5Instance = editor.p5Instance;
      const textWidth = p5Instance.textWidth(editor.commandInput);
      const statusBarHeight = 24;
      const editorHeight = p5Instance.height - statusBarHeight;
      const statusY = editorHeight;
      
      editor.hiddenInput.style.left = `${10 + textWidth}px`;
      editor.hiddenInput.style.top = `${statusY + 3}px`;
    }
  }
  
  private dispatchCommandEvent(editor: IVimEditor, command: string): void {
    editor.emitCommand(command);
  }
}

