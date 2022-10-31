<script setup lang="ts">
import Textarea from 'primevue/textarea';
import Button from 'primevue/button';
import Datatable from 'primevue/datatable';
import Column from 'primevue/column';
import Dropdown from 'primevue/dropdown';
import InputText from 'primevue/inputtext';
import Checkbox from 'primevue/checkbox';

import TabView from 'primevue/tabview';
import TabPanel from 'primevue/tabpanel';

import { reactive, ref } from 'vue';
import '@/models/csv-parser';
import { ClassProperty, CsvStringToInt32, CsvStringToString, VarType, type ICodeConverter, type IDataConverterData } from '@/typings/convert-models';

// defineProps<{
//   msg: string
// }>()

let data = reactive<IDataConverterData>({
  sourceCsvText: `id,name\n1,flash\n2,jack`,
  sourceObjArrJson: `[{"id":"1","name":"flash"},{"id":"2","name":"jack"}]`,
  sourceLine: `1,2,3,4`,
  className: "MyClass",
  targetProperties: [],
  targetText1: "",
  targetText2: "",
  targetText3: "",
  lines: [],
  isCamelCase: true,
  separator: ',',
  templateText: "name: ${this.id}",
  isAddBreak: true,
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

function fillTemplate(templateString: string, templateVars: any) {
  return new Function("return `" + templateString + "`;").call(templateVars);
}

function onConvertLineToLines() {
  let lines = data.sourceLine.csvSplit(data.separator);
  lines = lines.filter(x => x.trim() != "");
  data.targetText2 = lines.join('\n');
}

function dataLinesToJson(columns: ClassProperty[], lines: string[]) {
  let result: object[] = [];
  lines.forEach((line) => {
    let obj: any = {};
    line.csvSplit().forEach((value, idx) => {
      let name = camelCase(columns[idx].name);
      obj[name] = value;
    });
    result.push(obj);
  });
  return JSON.stringify(result);
}

function objArrayJsonToText(objArrayJsonString: string): string {
  let objArray: any[] = JSON.parse(objArrayJsonString);
  let result = "";
  objArray.forEach((obj) => {
    let item = fillTemplate(data.templateText, obj);
    result += item;
    if (data.isAddBreak) {
      result += "\n";
    }
  });
  return result;
}

function onConvertObjArrayJsonToTemplate() {
  let template = 'new MyClass {\n';
  let objArray: any[] = JSON.parse(data.sourceObjArrJson);
  let keys = Object.keys(objArray[0]);
  keys.forEach((key, idx) => {
    template += `\t${key} = "\$\{this.${key}\}"`;
    if (idx < keys.length - 1) {
      template += ',';
    }
    template += '\n';
  });
  template += '},';
  data.templateText = template;
}

function onConvertObjArrayJsonToText() {
  data.targetText3 = objArrayJsonToText(data.sourceObjArrJson);
}

function onConvertToJson() {
  let lines = data.sourceCsvText.csvSplit('\n');
  convertToColumns(lines);
  data.lines = lines.slice(1);
  data.targetText1 = dataLinesToJson(data.targetProperties, data.lines);
}

function dataLinesToClass(columns: ClassProperty[], dataLines: string[]) {
  let result = "";
  dataLines.forEach((line, index) => {
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

    if (index < dataLines.length - 1) {
      code += ",";
    }
    code += "\r\n"
    result += code;
  });
  return result;
}

function getColumns(names: string[]) {
  let result: ClassProperty[] = [];
  names.forEach((name, idx) => {
    result.push(new ClassProperty({
      name: name,
    }));
  });
  return result;
}

function convertToColumns(lines: string[]) {
  let columnNames: string[] = lines.getCsvHeaders();
  data.targetProperties = getColumns(columnNames);
}

function onConvertToClass() {
  let lines = data.sourceCsvText.csvSplit('\n');
  convertToColumns(lines);

  data.lines = lines.slice(1);
  data.targetText1 = dataLinesToClass(data.targetProperties, data.lines);
}

function onRefreshToClass() {
  data.targetText1 = dataLinesToClass(data.targetProperties, data.lines);
}

function onRefreshToJson() {
  data.targetText1 = dataLinesToJson(data.targetProperties, data.lines);
}
</script>

<template>
  <div>
    <TabView>
      <TabPanel header="Csv To Class">
        <div class="mb-3">
          <Textarea v-model="data.sourceCsvText" rows="10" cols="80"></Textarea>
        </div>
        <div class="mb-3">
          <Button label="ToClass" :onclick="onConvertToClass"></Button>
          &nbsp;
          <Button label="ToJson" :onclick="onConvertToJson"></Button>
        </div>
        <div>
          Class Name
          <InputText type="text" v-model="data.className" />
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
          <div class="mb-3">
            <Textarea v-model="data.targetText1" rows="10" cols="80"></Textarea>
          </div>
        </div>
      </TabPanel>
      <TabPanel header="Line To Lines">
        <div class="mb-3">
          <Textarea v-model="data.sourceLine" rows="2" cols="80"></Textarea>
        </div>
        <div class="mb-3">
          separator
          <InputText type="text" v-model="data.separator" />
          &nbsp;
          <Button :onclick="onConvertLineToLines">line to lines</Button>
        </div>
        <div class="mb-3">
          <Textarea v-model="data.targetText2" rows="10" cols="80"></Textarea>
        </div>
      </TabPanel>
      <TabPanel header="Obj Array Json To Text">
        <div class="mb-3">
          <Textarea v-model="data.sourceObjArrJson" rows="10" cols="80"></Textarea>
        </div>
        <div class="mb-3">
          <Button :onclick="onConvertObjArrayJsonToText">Obj Array Json to Text</Button>
          &nbsp;
          <Checkbox v-model="data.isAddBreak" :binary="true" /> Add Break
          &nbsp;
          <Button :onclick="onConvertObjArrayJsonToTemplate">Generate Class Template</Button>
        </div>
        <div class="mb-3">
          <Textarea v-model="data.templateText" rows="5" cols="80"></Textarea>
        </div>
        <div class="mb-3">
          <Textarea v-model="data.targetText3" rows="10" cols="80"></Textarea>
        </div>
      </TabPanel>
    </TabView>
  </div>
</template>

<style scoped>

</style>
