<script setup lang="ts">
import Textarea from 'primevue/textarea';
import Button from 'primevue/button';
import Datatable from 'primevue/datatable';
import Column from 'primevue/column';
import Dropdown from 'primevue/dropdown';
import InputText from 'primevue/inputtext';
import Checkbox from 'primevue/checkbox';

import { reactive, ref } from 'vue';
import '@/models/csv-parser';
import { ClassProperty, CsvStringToInt32, CsvStringToString, VarType, type ICodeConverter, type IDataConverterData } from '@/typings/convert-models';

// defineProps<{
//   msg: string
// }>()

let data = reactive<IDataConverterData>({
  sourceText: "",
  className: "MyClass",
  targetProperties: [],
  targetText: "",
  lines: [],
  isCamelCase: true,
});

let varTypes = ref([
  { text: 'String', value: VarType.String },
  { text: 'Int32', value: VarType.Int32 },
]);

let codeConverter: Record<VarType, ICodeConverter> = {
  [VarType.String]: new CsvStringToString(),
  [VarType.Int32]: new CsvStringToInt32(),
};

function camelCase(text: string) {
  if (!data.isCamelCase) {
    return text;
  }
  return text.substring(0, 1).toUpperCase() + text.substring(1);
}

function linesToJson(columns: ClassProperty[], lines: string[])
{
  let result: object[] = [];
  lines.forEach((line, index) => {
    let obj: any = {};
    line.csvSplit().forEach((elem, idx) => {
      let name = columns[idx].name;
      obj[name] = elem;
    });
    result.push(obj);
  });
  console.log('aaa', result);
  return JSON.stringify(result);
}

function convertToJson() {
  let lines = data.sourceText.csvSplit('\n');
  getColumns(lines);
  data.lines = lines.slice(1);
  data.targetText = linesToJson(data.targetProperties, data.lines);
}

function linesToClass(columns: ClassProperty[], lines: string[]) {
  let result = "";
  lines.forEach((line, index) => {
    let code = "";
    code += `new ${data.className} { \r\n`;
    line.csvSplit().forEach((elem, idx) => {
      let column = columns[idx];
      let propertyName = camelCase(column.name);
      code += `\t${propertyName} = ${codeConverter[column.type].to(elem)}`;
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
  return result
}

function getColumns(lines: string[]) {
  let columns: string[] = [];
  columns = lines.getCsvHeaders();
  data.targetProperties = [];
  columns.forEach((name, idx) => {
    data.targetProperties.push(new ClassProperty({
      name: name,
    }));
  });
}

function convertToClass() {
  let lines = data.sourceText.csvSplit('\n');
  getColumns(lines);

  data.lines = lines.slice(1);
  data.targetText = linesToClass(data.targetProperties, data.lines);
}

function onRefreshToClass() {
  data.targetText = linesToClass(data.targetProperties, data.lines);
}

function onRefreshToJson() {
  data.targetText = linesToJson(data.targetProperties, data.lines);
}
</script>

<template>
  <div>
    <div class="mb-3">
      <Textarea v-model="data.sourceText" rows="10" cols="80"></Textarea>
    </div>
    <div>
      Class Name <InputText type="text" v-model="data.className" />
      <Checkbox v-model="data.isCamelCase" :binary="true" /> CamelCase
    </div>
    <div>
      <Datatable :value="data.targetProperties" responsive-layout="scroll">
        <template #header>
          <div class="table-header">
            <Button :onclick="onRefreshToClass">Refresh Class</Button>
            &nbsp;
            <Button :onclick="onRefreshToJson">Refresh Json</Button>
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
      <Button label="ToClass" :onclick="convertToClass"></Button>
      &nbsp;
      <Button label="ToJson" :onclick="convertToJson"></Button>
    </div>
  </div>
</template>

<style scoped>

</style>
