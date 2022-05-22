import { NInput } from "naive-ui";
import { defineComponent, h, nextTick, PropType, ref } from "vue";

export default defineComponent({
  props: {
    value: { type: String, required: true },
    onUpdateValue: {
      type: Function as PropType<(value: string) => void>,
      required: false,
    },
  },
  setup(props) {
    const isEdit = ref(false);
    const inputRef = ref<HTMLElement>();
    const inputValue = ref(props.value);

    function handleOnClick() {
      isEdit.value = true;
      nextTick(() => {
        inputRef.value!.focus();
      });
    }

    function handleChange() {
      let onUpdateValueFn = props.onUpdateValue as Function;
      if (onUpdateValueFn != null) {
        onUpdateValueFn(inputValue.value);
      }
      isEdit.value = false;
    }

    return () => (
      <div onClick={handleOnClick}>
        {isEdit.value ? (
          <NInput
            ref={inputRef}
            v-model:value={inputValue.value}
            onUpdateValue={(v) => {
              inputValue.value = v;
            }}
            onChange={handleChange}
            onBlur={handleChange}
          />
        ) : (
          props.value
        )}
      </div>
    );
  },
});
