import { ComponentPublicInstance, defineComponent, onMounted, reactive, ref } from "vue";

export interface IPreviewFrameExpose {
  getContent(): string;
}

export interface IPreviewFrameProxy extends ComponentPublicInstance, IPreviewFrameExpose {}

export default defineComponent({
  props: {
    content: { type: String, required: true },
  },
  setup(props, { expose, slots }) {
    const state = reactive({
      content: props.content,
    });

    const iframe = ref<HTMLIFrameElement>();

    const getContent = () => {
      return state.content;
    };
    expose({
      getContent,
    } as IPreviewFrameExpose);

    return () => (
      <iframe src="" width="100%" height="320" frameborder="1" 
        style="position:relative; z-index:999"
        srcdoc={state.content}
        sandbox="allow-scripts allow-same-origin"
        ref={iframe}>
        your browser not support iframe.
      </iframe>
    );
  },
});

