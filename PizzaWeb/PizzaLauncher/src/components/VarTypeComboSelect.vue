<template>
  <AutoComplete 
    v-model="state.varType" 
    :suggestions="state.filteredVarTypes" 
    @itemSelect="handleSelectVarType($event)"
    @complete="handleSearchVarType($event)"
    field="label" 
    :dropdown="true" />
</template>

<script setup lang="ts">
import { AutoCompleteCompleteEvent, AutoCompleteItemSelectEvent } from "primevue/autocomplete";
import {
  ComponentPublicInstance,
  defineComponent,
  onMounted,
  PropType,
  reactive,
  ref,
} from "vue";
import { DefaultTemplateVariableOptions, IOption } from "@/typings/ui-typeings";
import AutoComplete from "primevue/autocomplete";

const props = defineProps({
  modelValue: {
    type: String,
    default: () => {
      return "String";
    },
  },
});

const state = reactive({
  varType: props.modelValue,
  filteredVarTypes: [] as IOption[],
  inputValue: "",
});


const emit = defineEmits([
  'update:modelValue'
]);

const handleBlur = () => {
  console.log("b", state.varType);
};

function handleSearchVarType(event: AutoCompleteCompleteEvent) {
  setTimeout(() => {
    if (!event.query.trim().length) {
      state.filteredVarTypes = [...DefaultTemplateVariableOptions];
    }
    else {
      state.filteredVarTypes = DefaultTemplateVariableOptions.filter((item) => {
          return item.label.toLowerCase().startsWith(event.query.toLowerCase());
        });
    }
    //emit('update:modelValue', state.varType);
  }, 250);
}

function handleSelectVarType(event: AutoCompleteItemSelectEvent) {
  state.varType = event.value.label;
  emit('update:modelValue', state.varType);
}
</script>