<template>
  <div>
    <div class="btn-group">
    </div>
    <div class="content-area">
      <div class="visuell-view" contenteditable 
        v-show="state.isVisuell"></div>
      <Textarea class="html-view" v-show="!state.isVisuell" 
        v-model="state.content"
        @blur="handleBlur"
        rows="10" cols="100"></Textarea>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ComponentPublicInstance,
  defineComponent,
  onMounted,
  reactive,
  ref,
} from "vue";
import "./Editor.scss";
import Textarea from 'primevue/textarea';

export interface IEditorExpose {
  getContent(): string;
}

export interface IEditorProxy extends ComponentPublicInstance, IEditorExpose { }

const props = defineProps({
  modelValue: { type: String, required: true },
});

const emit = defineEmits([
  'update:modelValue'
]);

const state = reactive({
  content: props.modelValue,
  isVisuell: false,
});


function handleBlur() {
  console.log("blur");
  emit('update:modelValue', state.content);
}

const getContent = () => {
  return state.content;
};

defineExpose({
  getContent,
} as IEditorExpose);

</script>