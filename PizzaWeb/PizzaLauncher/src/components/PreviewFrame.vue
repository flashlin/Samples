<template>
  <iframe name="preview" :src="props.src"
    width="100%" height="320" 
    :style="{border: '2px solid #ccc;'}"
    :srcdoc="props.content"
    sandbox="allow-scripts allow-same-origin" 
    @ref="iframe">
    your browser not support iframe.
  </iframe>
</template>

<script setup lang="ts">
import {
  ComponentPublicInstance,
  onMounted,
  reactive,
  ref,
} from "vue";

export interface IPreviewFrameExpose {
  getContent(): string;
  reload(): void;
}

export interface IPreviewFrameProxy
  extends ComponentPublicInstance,
  IPreviewFrameExpose { }

const props = defineProps({
  src: { type: String, required: false },
  content: { type: String, required: true },
});

const state = reactive({
  src: props.src == null ? "" : props.src,
  content: props.content,
});

const iframe = ref<HTMLIFrameElement>();

const getContent = () => {
  return state.content;
};

const reload = () => {
  if (iframe.value == null) {
    return;
  }
  iframe.value.contentWindow?.location.reload();
};

defineExpose({
  getContent,
  reload,
} as IPreviewFrameExpose);

</script>