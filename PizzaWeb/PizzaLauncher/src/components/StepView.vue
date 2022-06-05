<template>
   <component 
      :is="`Step${state.currentStep}`"
      @change-step="changeStep"
   />
</template>

<script setup lang="ts">
import { reactive } from 'vue';

const state = reactive({
  currentStep: 0,
});

function changeStep(n: number) {
   state.currentStep = n;
}
</script>