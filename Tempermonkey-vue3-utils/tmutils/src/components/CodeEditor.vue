<template>
   <div class="editor" ref="dom"></div>
</template>

<script setup lang="ts">
import * as monaco from 'monaco-editor';
import { reactive, ref, onMounted } from 'vue';
import axios from 'axios';

import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';


const props = defineProps<{
   modelValue: string,
}>();

const dom = ref();

const emit = defineEmits<{
   (e: 'update:modelValue', value: string): void
}>()

// @ts-ignore
self.MonacoEnvironment = {
   getWorker(workerId: string, label: string) {
      if (label === 'json') {
         return new JsonWorker();
      }
      return new EditorWorker();
   },
};

let instance: monaco.editor.IStandaloneCodeEditor;

const createTypescriptDefinitionModel = async () => {
   const libUri = "ts:filename/index.d.ts";
   const libSource = (await axios.get("index.d.ts")).data;
   monaco.languages.typescript.javascriptDefaults.addExtraLib(libSource, libUri);
   monaco.editor.createModel(libSource, 'typescript', monaco.Uri.parse(libUri));
};

onMounted(() => {
   // add support typescript
   monaco.languages.typescript.javascriptDefaults.setDiagnosticsOptions({
      noSemanticValidation: true,
      noSyntaxValidation: false,
   });
   monaco.languages.typescript.javascriptDefaults.setCompilerOptions({
      target: monaco.languages.typescript.ScriptTarget.ES2016,
   });


   const jsonModel = monaco.editor.createModel(
      props.modelValue,
      'json',
      monaco.Uri.parse('json://grid/settings.json')
   );

   instance = monaco.editor.create(dom.value, {
      model: jsonModel,
      tabSize: 2,
      automaticLayout: true,
      scrollBeyondLastLine: false,
   });

   instance.onDidChangeModelContent(() => {
      const value = instance.getValue();
      emit('update:modelValue', value);
   });
});
</script>

<style  scoped>
.editor {
   width: 600px;
   height: 800px;
}
</style>