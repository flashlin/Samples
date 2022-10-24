<template>
  <div>
    <Textarea v-model="data.sourceText" rows="10" cols="80"></Textarea>
    <Textarea v-model="data.targetText" rows="20" cols="80"></Textarea>
    <Button label="ToClass"></Button>
  </div>
</template>

<script setup lang="ts">
import Textarea from 'primevue/textarea';
import Button from 'primevue/button';
import { ref } from 'vue';

// defineProps<{
//   msg: string
// }>()

let data = ref({
  sourceText: "",
  targetText: ""
});

</script>

<style scoped>
</style>
