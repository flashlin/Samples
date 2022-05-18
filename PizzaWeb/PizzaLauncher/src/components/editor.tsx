import {
  ComponentPublicInstance,
  defineComponent,
  onMounted,
  reactive,
  ref,
} from "vue";
import "./Editor2.scss";

export interface IEditorExpose {
  getContent(): string;
}

export interface IEditorProxy extends ComponentPublicInstance, IEditorExpose {}

export default defineComponent({
  props: {
    content: { type: String, required: true },
  },
  setup(props, { expose, slots }) {
    const state = reactive({
      content: props.content,
      isVisuell: false,
    });

    if (!state.content) {
      state.content = "";
    }

    const getContent = () => {
      return state.content;
    };

    expose({
      getContent,
    } as IEditorExpose);

    return () => (
      <div>
        <div class="btn-group">
          <a href="#" class="btn btn-primary active" aria-current="page">
            Active link
          </a>
          <a href="#" class="btn btn-primary">
            Link
          </a>
          <a href="#" class="btn btn-primary">
            Link
          </a>
        </div>
        <div class="content-area">
          <div class="visuell-view" contenteditable v-show={state.isVisuell}></div>
          <textarea class="html-view" v-show={!state.isVisuell} v-model={state.content}></textarea>
        </div>
      </div>
    );
  },
});
