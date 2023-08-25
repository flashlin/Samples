<script setup lang="ts">
import { reactive, shallowRef } from 'vue';
import { Codemirror } from 'vue-codemirror';
import { sql } from '@codemirror/lang-sql';
import { oneDark } from '@codemirror/theme-one-dark';
import { CompletionContext, startCompletion } from "@codemirror/autocomplete";

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

const codeEditorRef = shallowRef();
const data = reactive({
  code: props.modelValue,
});

const extensions = [sql(), oneDark];

interface ICodemirrorPayload {
  view: import('@codemirror/view').EditorView;
}

const customAutocompleteList = ['apple', 'banana', 'cherry', 'date', 'elderberry'];
function customHint(editor: any) {
  const cursor = editor.getCursor();
  const currentLine = editor.getLine(cursor.line);
  const currentWord = currentLine.slice(0, cursor.ch);

  // 過濾出符合的自動完成項目
  const suggestions = customAutocompleteList.filter(item =>
    item.startsWith(currentWord)
  );

  // 返回自動完成建議物件
  return {
    list: suggestions,
    from: Codemirror.Pos(cursor.line, currentWord.length),
    to: Codemirror.Pos(cursor.line, cursor.ch)
  };
}


const handleReady = (payload: ICodemirrorPayload) => {
  codeEditorRef.value = payload.view;
  console.log('read', codeEditorRef.value)
  //startCompletion(codeEditor.value);
};

const handleChange = (value: string, viewUpdate: import("@codemirror/view").ViewUpdate) => {
  console.log('change', viewUpdate);
  const editor = codeEditorRef.value
  console.log('change2', editor);

  const cursor = getCursor();

  const hintOptions = {
    hint: () => ({
      from: Codemirror.Pos(cursor.line, 0),
      to: Codemirror.Pos(cursor.line, cursor.ch),
      list: ['HELLO']
    })
  };

  editor.showHint(hintOptions);


  emits('update:modelValue', value);
};

const myCompletions = (context: CompletionContext) => {
  console.log('trigger', context)
  let word = context.matchBefore(/\w*/);
  if (!word) return null;
  if (word.from == word.to && !context.explicit)
    return null;
  return {
    from: word.from,
    options: [
      { label: "match", type: "keyword" },
      { label: "hello", type: "variable", info: "(World)" },
      { label: "magic", type: "text", apply: "⠁⭒*.✩.*⭒⠁", detail: "macro" }
    ]
  }
};

const getCursor = () => {
  const state = codeEditorRef.value.viewState.state;
  const ranges = state.selection.ranges;
  const selected = ranges.reduce((r: number, range: any) => r + range.to - range.from, 0);
  const cursorPos = ranges[0].anchor;
  return cursorPos;
}

const getInfo = () => {
  const state = codeEditorRef.value.viewState.state;
  const ranges = state.selection.ranges;
  const selected = ranges.reduce((r: number, range: any) => r + range.to - range.from, 0);
  const cursor = ranges[0].anchor;
  console.log('ranges', ranges)
  //const length = state.doc.length
  //const lines = state.doc.lines
  console.log('get', selected, cursor)
  return;
};



interface ICodeEditorExpose {
  getInfo(): any;
}
defineExpose<ICodeEditorExpose>({
  getInfo,
});
</script>

<template>
  <codemirror v-model="data.code" placeholder="Code goes here..." :style="{ height: '400px' }" :autofocus="true"
    :indent-with-tab="true" :tab-size="2" :extensions="extensions" @ready="handleReady" @change="handleChange" />
</template>
