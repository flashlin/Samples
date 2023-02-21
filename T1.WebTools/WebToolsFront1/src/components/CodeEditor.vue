<script setup lang="ts">
import { reactive } from 'vue';
import CodeMirror from 'vue-codemirror6';
import {defaultKeymap} from "@codemirror/commands"
import {keymap} from "@codemirror/view"

const data = reactive({
    code: "",
});
const extensions =  [keymap.of(defaultKeymap)];

function onChange(val: string, cm: any) {

}
</script>


<template>
    <code-mirror v-model="data.code" :extensions="extensions" />
</template>