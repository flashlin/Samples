<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { Terminal } from 'xterm';
import 'xterm/css/xterm.css';
import { WebLinksAddon } from 'xterm-addon-web-links';
import type { ITerminalUiExpose } from '@/components/TerminalUiModel';

const terminalDom = ref<HTMLElement>();
const terminal = new Terminal();
terminal.loadAddon(new WebLinksAddon());

function executeCommand(command: any) {
   const terminal: any = terminalDom.value!;
   terminal.writeln(`> ${command}`);
}

function writeln(text: string): void {
   terminal.writeln(text);
}

defineExpose<ITerminalUiExpose>({
   writeln,
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
      <div ref="terminalDom" style="height: 50px;"></div>
   </div>
</template>

<style scoped>
@import "xterm/css/xterm.css";
</style>
