import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class CommandModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.Command;
  
  onEnter(editor: IVimEditor): void {
    console.log('CommandModeHandler.onEnter called');
    editor.commandInput = ':';
    console.log('hiddenInput exists:', !!editor.hiddenInput);
    if (editor.hiddenInput) {
      const p5Instance = editor.p5Instance;
      if (p5Instance) {
        const statusBarHeight = 24;
        const editorHeight = p5Instance.height - statusBarHeight;
        const statusY = editorHeight;
        
        editor.hiddenInput.style.left = '10px';
        editor.hiddenInput.style.top = `${statusY + 3}px`;
      }
      
      console.log('Focusing hiddenInput...');
      editor.hiddenInput.focus();
      console.log('hiddenInput focused, activeElement:', document.activeElement);
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
          this.dispatchCommandEvent(editor, command);
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
          }
        } 
      },
    ];
  }
  
  handleKey(key: string, editor: IVimEditor): void {
    console.log('CommandModeHandler.handleKey called with key:', key);
    const keyPatterns = this.getKeyPatterns(editor);
    
    for (const { pattern, action } of keyPatterns) {
      if (pattern.test(key)) {
        console.log('Pattern matched:', pattern);
        action();
        if (editor.p5Instance) {
          editor.p5Instance.redraw();
        }
        return;
      }
    }
    console.log('No pattern matched for key:', key);
  }
  
  handleInput(editor: IVimEditor, value: string): void {
    console.log('CommandModeHandler.handleInput called with value:', value);
    console.log('Current commandInput:', editor.commandInput);
    editor.commandInput += value;
    console.log('Updated commandInput:', editor.commandInput);
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  handleCompositionEnd(editor: IVimEditor, data: string): void {
    if (data) {
      editor.commandInput += data;
      if (editor.p5Instance) {
        editor.p5Instance.redraw();
      }
    }
  }
  
  private dispatchCommandEvent(editor: IVimEditor, command: string): void {
    const event = new CustomEvent('vim-command', {
      detail: { command },
      bubbles: true,
      composed: true
    });
    editor.dispatchEvent(event);
  }
}

