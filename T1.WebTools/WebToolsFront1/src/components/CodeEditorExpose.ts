import type { ComponentPublicInstance } from "vue";

export interface ICodeEditorExpose {
  getSelectionCodeText(): string;
}

export interface ICodeEditorProxy
  extends ComponentPublicInstance,
    ICodeEditorExpose {}
