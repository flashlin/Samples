import type { ComponentPublicInstance } from "vue";

export interface ITerminalUiExpose {
  write(text: string): void;
  writeln(text: string): void;
  clear(): void;
}

export interface ITerminalUiProxy
  extends ComponentPublicInstance,
    ITerminalUiExpose {}
