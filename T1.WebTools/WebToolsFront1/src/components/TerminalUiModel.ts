import type { ComponentPublicInstance } from "vue";

export interface ITerminalUiExpose {
  writeln(text: string): void;
  clear(): void;
}

export interface ITerminalUiProxy
  extends ComponentPublicInstance,
    ITerminalUiExpose {}
