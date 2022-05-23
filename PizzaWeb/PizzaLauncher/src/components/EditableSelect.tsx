import { NButton, NDropdown, NInput, NSelect } from "naive-ui";
import {
  ComponentPublicInstance,
  defineComponent,
  onMounted,
  PropType,
  reactive,
  ref,
} from "vue";
import "./Editor.scss";

export class EditableSelectOption {
  constructor(data?: Partial<EditableSelectOption>) {
    Object.assign(this, data);
  }
  label: string = "";
  key: string = "";
  disabled: boolean = false;
}

export interface IEditableSelectExpose {
  getContent(): string;
}

export interface IEditableSelectProxy
  extends ComponentPublicInstance,
    IEditableSelectExpose {}

export default defineComponent({
  props: {
    value: { type: String, required: true },
    options: {
      type: Object as PropType<EditableSelectOption[]>,
      required: true,
    },
  },
  setup(props, { expose, slots }) {
    const state = reactive({
      value: props.value,
      options: props.options,
      inputValue: "",
      isShowList: false,
    });

    console.log("options", state.options);

    const showDropdownList = (show: boolean) => {
      state.isShowList = show;
    };

    const handleSelect = (key: string) => {
      console.log("select key", key);
    };

    const getContent = () => {
      return state.value;
    };

    expose({
      getContent,
    } as IEditableSelectExpose);

    return () => (
      <div class="autocomplete">
        <NInput
          onFocus={() => showDropdownList(true)}
          onBlur={() => showDropdownList(false)}
          value={state.inputValue}
        />
        <NSelect v-model:options={state.options} onChange={handleSelect} filterable>
        </NSelect>
      </div>
    );
  },
});
