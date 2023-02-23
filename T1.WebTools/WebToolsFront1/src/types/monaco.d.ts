import { Environment } from 'monaco-editor/esm/vs/editor/editor.api';

declare global {
    interface Window {
      MonacoEnvironment: Environment;
    }
}