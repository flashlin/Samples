import type { ComponentPublicInstance } from "vue";

export interface ITerminalUiExpose {
  writeln(text: string): void;
}

export interface ITerminalUiProxy
  extends ComponentPublicInstance,
    ITerminalUiExpose {}
