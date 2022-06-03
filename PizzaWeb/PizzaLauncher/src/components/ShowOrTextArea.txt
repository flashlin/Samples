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
      if (props.onUpdateValue != null) {
        props.onUpdateValue(inputValue.value);
      }
      isEdit.value = false;
    }

    return () => (
      <div onClick={handleOnClick}>
        {isEdit.value ? (
          <NInput
            type="textarea"
            rows="10"
            ref={inputRef}
            v-model:value={inputValue.value}
            onChange={handleChange}
            onBlur={handleChange}
          />
        ) : (
          <span>{props.value}</span>
        )}
      </div>
    );
  },
});
