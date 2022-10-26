<script setup lang="ts">
import Textarea from 'primevue/textarea';
import Button from 'primevue/button';
import Datatable from 'primevue/datatable';
import Column from 'primevue/column';
import Dropdown from 'primevue/dropdown';
import InputText from 'primevue/inputtext';

import { reactive, ref } from 'vue';
import '@/models/csv-parser';
import { ClassProperty, VarType, type IDataConverterData } from '@/typings/convert-models';

// defineProps<{
//   msg: string
// }>()

let data = reactive<IDataConverterData>({
  sourceText: "",
  className: "MyClass",
  targetProperties: [],
  targetText: "",
});

let varTypes = ref([
  { text: 'String', value: VarType.String },
  { text: 'Int32', value: VarType.Int32 },
]);

function convertToJson() {
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

function convertToClassValues() {
  let columns: string[] = [];
  let result = "";
  let lines = data.sourceText.csvSplit('\n');

  columns = lines.getCsvHeaders();
  data.targetProperties = [];
  columns.forEach((name, idx) => {
    data.targetProperties.push(new ClassProperty({
      name: name,
      type: VarType.String
    }));
  });

  lines.slice(1).forEach((line, index) => {
    let code = "";
    code += `new ${data.className} { \r\n`;
    line.csvSplit().forEach((elem, idx) => {
      let name = columns[idx];
      code += `${name} = ${elem}`;
      if (idx < columns.length - 1) {
        code += ',';
      }
      code += '\r\n';
    });
    code += "}";

    if (index < lines.length - 1) {
      code += ",";
    }
    code += "\r\n"
    result += code;
  });
  data.targetText = result;
}
</script>

<template>
  <div>
    <div class="mb-3">
      <Textarea v-model="data.sourceText" rows="10" cols="80"></Textarea>
    </div>
    <div>
      <InputText type="text" v-model="data.className" />
    </div>
    <div>
      <Datatable :value="data.targetProperties" responsive-layout="scroll">
        <template #header>
          <div class="table-header">
            Name
            <Button icon="pi pi-refresh"></Button>
          </div>
        </template>
        <Column field="name" header="Name"></Column>
        <Column field="type" header="Type">
          <template #body="slotProps">
            <Dropdown v-model="slotProps.data.type" :options="varTypes" optionLabel="text" optionValue="value"
              :filter="true" placeholder="Select a VarType" :showClear="false">
              <!-- <template #value="slotProps">
                <div v-if="slotProps.value">
                  <div>{{ slotProps.value.text }}</div>
                </div>
                <span v-else>
                  {{ slotProps.placeholder }}
                </span>
              </template>
              <template #option="slotProps">
                <div class="country-item">
                  <div>{{ slotProps.option.name }}</div>
                </div>
              </template> -->
            </Dropdown>
          </template>
        </Column>
        <template #footer>
        </template>
      </Datatable>
    </div>
    <div class="mb-3">
      <Textarea v-model="data.targetText" rows="10" cols="80"></Textarea>
    </div>
    <div class="mb-3">
      <Button label="ToClass" :onclick="convertToClassValues"></Button>
    </div>
  </div>
</template>

<style scoped>

</style>
