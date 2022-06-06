<template>
  <div>
    <AutoComplete v-model="state.modelValue" :suggestions="state.filteredOptions"
      @complete="handleSearchOptions($event)" 
      field="label"
      @itemSelect="handleSelect" :dropdown="true" />
  </div>

</template>

<script setup lang="ts">
import {
  ComponentPublicInstance,
  onMounted,
  PropType,
  reactive,
  ref,
} from "vue";
import AutoComplete, { AutoCompleteCompleteEvent, AutoCompleteItemSelectEvent } from "primevue/autocomplete";
import InputText from "primevue/inputtext";
import { IOption } from "@/typings/ui-typeings";

type GetOptionsFunc = () => Promise<IOption[]>;

const props = defineProps({
  modelValue: {
    type: String,
    default: () => {
      return "";
    },
  },
  options: {
    type: [Array, Function] as PropType<Array<IOption> | GetOptionsFunc>,
    default: () => {
      return [] as IOption[];
    },
  },
});

const state = reactive({
  modelValue: props.modelValue,
  filteredOptions: [] as IOption[],
});

const emit = defineEmits([
  'update:modelValue'
]);

async function getOptionsList() {
  if( typeof props.options == "function") {
    return await props.options();
  }
  return props.options;
}

function handleSearchOptions(event: AutoCompleteCompleteEvent) {
  setTimeout(async () => {
    let optionsList = await getOptionsList();
    if (!event.query.trim().length) {
      state.filteredOptions = [...optionsList];
    }
    else {
      state.filteredOptions = optionsList.filter((item) => {
        return item.label.toLowerCase().startsWith(event.query.toLowerCase());
      });
    }
    //emit('update:modelValue', state.modelValue);
  }, 250);
}

function handleSelect(event: AutoCompleteItemSelectEvent) {
  state.modelValue = event.value.value;
  emit('update:modelValue', state.modelValue);
}
</script>