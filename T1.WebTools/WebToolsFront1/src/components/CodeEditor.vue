<script setup lang="ts">
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue';
import * as monaco from 'monaco-editor';

const data = reactive({
    code: "",
});

const editorDom = ref<HTMLElement>();
let editor: monaco.editor.IStandaloneCodeEditor;

onMounted(() => {
    editor = monaco.editor.create(editorDom.value!, {
        value: 'function hello() {\n\talert("Hello world!");\n}',
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