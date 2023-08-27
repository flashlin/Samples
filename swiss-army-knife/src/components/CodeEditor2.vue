<script setup lang="ts">
import { onBeforeUnmount, onMounted, reactive, ref } from 'vue';
//import * as monaco from 'monaco-editor';
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
import '../monacoEx/userWorker';
//import "monaco-editor/esm/vs/editor/contrib/find/findController";
import { SqlSnippets } from '@/monacoEx/Suggestions';

interface ICodeEditorProps {
  modelValue: string;
}
const props = withDefaults(defineProps<ICodeEditorProps>(), {
  modelValue: '',
});
// interface ICodeEditorEmits {
//   (e: 'update:modelValue', value: string): void;
// }
// const emits = defineEmits<ICodeEditorEmits>();
const data = reactive({
  code: props.modelValue,
});
const domRef = ref();
const completionItemProvider = ref<monaco.IDisposable>();

//const [editor, setEditor] = ref<monaco.editor.IStandaloneCodeEditor | null>(null);
//const editorRef = ref<monaco.editor.IStandaloneCodeEditor | null>(null);

//const customAutocompleteList = ['apple', 'banana', 'cherry', 'date', 'elderberry'];


// const handleChange = (value: string, viewUpdate: import("@codemirror/view").ViewUpdate) => {
//   emits('update:modelValue', value);
// };

const testDatabase = [{
  databaseName: `databaseName`,
  tableOptions: [
    {
      tableName: `tableName1`,
      tableComment: "tableComment1",
      fieldOptions: [
        {
          fieldName: "fieldName1",
          fieldComment: "fieldComment1",
          fieldType: "string",
          tableName: "tableName1",
          databaseName: `databaseName`,
        },
      ]
    },
  ],
}];


onMounted(() => {
  const sqlSnippets = new SqlSnippets(
    [],
    testDatabase
  );
  monaco.languages.registerCompletionItemProvider('sql', {
    //triggerCharacters: [" ", ".", ...props.triggerCharacters],
    provideCompletionItems: (
      model: monaco.editor.ITextModel,
      position: monaco.Position,
      _context: monaco.languages.CompletionContext,
      _token: monaco.CancellationToken): monaco.languages.ProviderResult<monaco.languages.CompletionList> => {
      return sqlSnippets.provideCompletionItems(model, position) as monaco.languages.ProviderResult<monaco.languages.CompletionList>;
    },
  });

  monaco.editor.create(domRef.value, {
    value: data.code,
    language: 'sql',
    tabSize: 2,
    automaticLayout: true,
    scrollBeyondLastLine: false,
    theme: 'vs-dark',
    selectOnLineNumbers: true,
    fontSize: 14,
    lineHeight: 30,
    contextmenu: false, //關閉右鍵
    suggestOnTriggerCharacters: true,
    acceptSuggestionOnCommitCharacter: false,
    suggestSelection: "first",
    fontFamily: "MONACO",
    folding: false,
    minimap: {
      enabled: false,
    },
  });
});

onBeforeUnmount(() => {
  completionItemProvider.value?.dispose();
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
</style>../monacoEx/userWorker@/monacoEx/Suggestions