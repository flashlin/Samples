import { ComponentPublicInstance, defineComponent, onMounted, reactive, ref } from "vue";
import { QuillEditor } from "@vueup/vue-quill";
import '@vueup/vue-quill/dist/vue-quill.snow.css';

export interface IEditorExpose {
  getContent(): string;
}

export interface IEditorProxy extends ComponentPublicInstance, IEditorExpose {}

export default defineComponent({
  components: {
    QuillEditor,
  },
  props: {
    content: { type: String, required: true },
  },
  setup(props, { expose, slots }) {
    const state = reactive({
      content: props.content,
    });

    const getContent = () => {
      return state.content;
    };
    expose({
      getContent,
    } as IEditorExpose);

    return () => <QuillEditor theme="snow" style="height: 320pt;" v-model:content={state.content} contentType="html" />;
  },
});

