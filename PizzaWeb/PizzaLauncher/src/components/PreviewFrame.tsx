import {
  ComponentPublicInstance,
  defineComponent,
  onMounted,
  reactive,
  ref,
} from "vue";

export interface IPreviewFrameExpose {
  getContent(): string;
}

export interface IPreviewFrameProxy
  extends ComponentPublicInstance,
    IPreviewFrameExpose {}

export default defineComponent({
  props: {
    content: { type: String, required: true },
  },
  setup(props, { expose, slots }) {
    const state = reactive({
      content: props.content,
    });

    const iframe = ref<HTMLIFrameElement>();

    const test = () => {
      console.log("test prop", props.content);
      console.log("test state", state.content);
    };

    const getContent = () => {
      return state.content;
    };
    expose({
      getContent,
    } as IPreviewFrameExpose);

    return () => (
      <iframe
        name="preview"
        src=""
        width="100%"
        height="320"
        style={`border: 2px solid #ccc;`}
        srcdoc={props.content}
        sandbox="allow-scripts allow-same-origin"
        ref={iframe}
      >
        your browser not support iframe.
      </iframe>
    );
  },
});
