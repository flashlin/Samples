<script setup lang="ts">
/// <reference types="monaco-editor" />
import { ref, reactive, onMounted, onBeforeUnmount, toRefs } from 'vue';
import * as monaco from 'monaco-editor';

import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
import type { ICodeEditorExpose } from './CodeEditorExpose';


self.MonacoEnvironment = {
    getWorker(workerId: string, label: string) {
        if (label === 'json') {
            return new JsonWorker();
        }
        return new EditorWorker();
    },
};



const props = defineProps<{
    code: string
}>();

interface MyComponentEmits {
    (e: 'update:code', code: string): void
}
const emits = defineEmits<MyComponentEmits>();

const data = reactive({
    code: props.code,
});

const editorDom = ref<HTMLElement>();
let editor: monaco.editor.IStandaloneCodeEditor;

function getSelectionCodeText(): string {
    const selection = editor.getSelection();
    if (selection) {
        const model = editor.getModel()!;
        const selectedText = model.getValueInRange(selection);
        return selectedText;
    }
    return editor.getValue();
}

defineExpose<ICodeEditorExpose>({
    getSelectionCodeText
});

onMounted(() => {
    //const jsonModel = monaco.editor.createModel(props.modelValue, 'json');
    editor = monaco.editor.create(editorDom.value!, {
        //model: jsonModel,
        value: data.code,
        language: 'sql'
    });

    editor.onDidChangeModelContent(() => {
        const value = editor.getValue();
        //emits('update:modelValue', value);
        emits('update:code', value);
    });
});

onBeforeUnmount(() => {
    editor.dispose();
});
</script>


<template>
    <div>
        <div ref="editorDom" style="width:100%; height: 500px;"></div>
    </div>
</template>