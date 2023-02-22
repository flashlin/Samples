<script setup lang="ts">
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue';
import * as monaco from 'monaco-editor';

import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
self.MonacoEnvironment = {
    getWorker(workerId, label) {
        if (label === 'json') {
            return new JsonWorker();
        }
        return new EditorWorker();
    },
};


const data = reactive({
    code: "",
});

const editorDom = ref<HTMLElement>();
let editor: monaco.editor.IStandaloneCodeEditor;

onMounted(() => {
    //const jsonModel = monaco.editor.createModel(props.modelValue, 'json');
    editor = monaco.editor.create(editorDom.value!, {
        //model: jsonModel,
        value: 'CREATE TABLE customer(\n[id] int IDENTITY(1,1),\n[name] nvarchar(50))\n',
        language: 'sql'
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