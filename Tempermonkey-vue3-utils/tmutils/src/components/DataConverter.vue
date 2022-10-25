<template>
  <div>
    <Textarea v-model="data.sourceText" rows="10" cols="80"></Textarea>
    <Textarea v-model="data.targetText" rows="20" cols="80"></Textarea>
    <Button label="ToClass" :onclick="onToClass"></Button>
  </div>
</template>

<script setup lang="ts">
import Textarea from 'primevue/textarea';
import Button from 'primevue/button';
import { reactive, ref } from 'vue';
import { csvToArray } from '@/models/csv-parser';

// defineProps<{
//   msg: string
// }>()

let data = reactive({
  sourceText: "",
  targetText: ""
});

function onToClass() {
  let columns: string[] = [];
  let result: object[] = [];
  data.sourceText.csvSplit('\n').forEach((line, index) => {
    if (index == 0) {
      line.csvSplit().forEach((name, fieldIdx) => {
        columns.push(name);
      });
      return;
    }
    let obj: any = {};
    line.csvSplit().forEach((elem, idx) => {
      let name = columns[idx];
      obj[name] = elem;
    });
    result.push(obj);
  });
  data.targetText = JSON.stringify(result);
}

</script>

<style scoped>

</style>
