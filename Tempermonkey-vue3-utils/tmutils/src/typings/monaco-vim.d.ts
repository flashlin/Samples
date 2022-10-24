module 'monaco-vim' {
   interface IVimInstance {
      dispose();
   }
   function initVimMode(instance: monaco.editor.IStandaloneCodeEditor, statusDiv: HTMLElement): IVimInstance; 
}