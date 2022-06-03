<template>
  <div class="autocomplete">
    <input @focus="showDropdownList(true)" 
      @blur="showDropdownList(false)" 
      value={state.inputValue} />
  </div>
</template>

<script setup lang="ts">
import {
  ComponentPublicInstance,
  defineComponent,
  onMounted,
  PropType,
  reactive,
  ref,
} from "vue";
import "./Editor.scss";

class EditableSelectOption {
  constructor(data?: Partial<EditableSelectOption>) {
    Object.assign(this, data);
  }
  label: string = "";
  key: string = "";
  disabled: boolean = false;
}

interface IEditableSelectExpose {
  getContent(): string;
}

// export interface IEditableSelectProxy
//   extends ComponentPublicInstance,
//     IEditableSelectExpose {}


const props = defineProps({
    value: { type: String, required: true },
    options: {
      type: Object as PropType<EditableSelectOption[]>,
      required: true,
    },
});

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

    defineExpose({
      getContent,
    } as IEditableSelectExpose);

</script>