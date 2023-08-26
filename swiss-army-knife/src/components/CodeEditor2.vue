<script setup lang="ts">
import { onMounted, reactive, ref, shallowRef } from 'vue';
//import * as monaco from 'monaco-editor';
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
import './userWorker';

interface ICodeEditorProps {
  modelValue: string;
}
const props = withDefaults(defineProps<ICodeEditorProps>(), {
  modelValue: '',
});
interface ICodeEditorEmits {
  (e: 'update:modelValue', value: string): void;
}
const emits = defineEmits<ICodeEditorEmits>();
const data = reactive({
  code: props.modelValue,
});
const domRef = ref();
//const [editor, setEditor] = ref<monaco.editor.IStandaloneCodeEditor | null>(null);
//const editorRef = ref<monaco.editor.IStandaloneCodeEditor | null>(null);

//const customAutocompleteList = ['apple', 'banana', 'cherry', 'date', 'elderberry'];


// const handleChange = (value: string, viewUpdate: import("@codemirror/view").ViewUpdate) => {
//   emits('update:modelValue', value);
// };

onMounted(() => {
  monaco.editor.create(domRef.value, {
    value: data.code,
    language: 'sql',
    tabSize: 2,
    automaticLayout: true,
    scrollBeyondLastLine: false,
    theme: 'vs-dark',
  });
});

// interface ICodeEditorExpose {
//   getInfo(): any;
// }
// defineExpose<ICodeEditorExpose>({
//   getInfo,
// });
</script>

<template>
  <div ref="domRef" class="Editor"></div>
</template>

<style scoped>
.Editor {
  width: 80vw;
  height: 50vh;
}
</style>