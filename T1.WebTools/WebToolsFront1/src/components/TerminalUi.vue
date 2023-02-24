<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { Terminal } from 'xterm';
import 'xterm/css/xterm.css';
import { WebLinksAddon } from 'xterm-addon-web-links';
import type { ITerminalUiExpose } from '@/components/TerminalUiModel';

const terminalDom = ref<HTMLElement>();
const terminal = new Terminal({
   fontFamily: 'Courier New, Courier, monospace, "思源黑體"',
   cols: 100,
   screenReaderMode: true,
});
terminal.loadAddon(new WebLinksAddon());

function executeCommand(command: any) {
   const terminal: any = terminalDom.value!;
   terminal.writeln(`> ${command}`);
}

function clear(): void {
   terminal.clear();
}

function write(text: string): void {
   terminal.write(text);
}

function writeln(text: string): void {
   //const textEncoder = new TextEncoder();
   //terminal.writeln(textEncoder.encode(text));
   terminal.writeln(text);
}

defineExpose<ITerminalUiExpose>({
   write,
   writeln,
   clear,
});

onMounted(() => {
   terminal.open(terminalDom.value!);
   //terminal.writeln('Hello from mxterm.js ');
   // setInterval(() => {
   //    terminal.writeln("" + new Date());
   // }, 1000);
})
</script>

<template>
   <div>
      <div ref="terminalDom"></div>
   </div>
</template>

<style scoped>
@import "xterm/css/xterm.css";
</style>
