<template>
  <AutoComplete 
    v-model="state.modelValue" 
    :suggestions="state.filteredOptions"
    @complete="handleSearchOptions($event)"
    field="label" :dropdown="true" />
</template>

<script setup lang="ts">
import { AutoCompleteCompleteEvent } from "primevue/autocomplete";
import {
  ComponentPublicInstance,
  onMounted,
  PropType,
  reactive,
  ref,
} from "vue";
import { IOption } from "@/typings/ui-typeings";

const props = defineProps({
  modelValue: {
    type: String,
    default: () => {
      return "";
    },
  },
  options: {
    type: Array as PropType<Array<IOption>>,
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

function handleSearchOptions(event: AutoCompleteCompleteEvent) {
  setTimeout(() => {
    if (!event.query.trim().length) {
      state.filteredOptions = [...props.options];
    }
    else {
      state.filteredOptions = props.options.filter((item) => {
        return item.label.toLowerCase().startsWith(event.query.toLowerCase());
      });
    }
    emit('update:modelValue', state.modelValue);
  }, 250);
}
</script>