<script setup lang="ts">
import { reactive, ref } from 'vue';
import { CodeSnippet, useCodeSnippetService } from './models';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import { it } from 'node:test';
//import ColumnGroup from 'primevue/columngroup'; 
//import Row from 'primevue/row';

const data = reactive({
  selectedIndex: -1,
  selectedRow: CodeSnippet.Empty,
});

let codeSnippetList = reactive<CodeSnippet[]>([]);

async function loadData() {
  let app = useCodeSnippetService();
  let list: CodeSnippet[] = await app.queryAsync('');
  for (let item of list) {
    codeSnippetList.push(item);
  }
  if (list.length > 0) {
    data.selectedIndex = 0;
    data.selectedRow = list[0];
  }
}

function rowClass(item: CodeSnippet) {
  let css = "";
  if (item.id == data.selectedRow.id) {
    css = 'row-selected';
  }
  console.log(`${item.id} ${css}`);
  return css;
}

function selectDown(event: KeyboardEvent) {
  if( event.key == 'ArrowDown' ) {
    data.selectedRow = codeSnippetList[data.selectedIndex + 1];
    data.selectedIndex++;
    return;
  }
  
  if( event.key == 'ArrowUp' && data.selectedIndex > 0) {
    data.selectedRow = codeSnippetList[data.selectedIndex - 1];
    data.selectedIndex--;
  }
}

document.addEventListener("keydown", selectDown)

loadData();

</script>

<template>
  <div v-on:keydown.down="selectRow()">
    <DataTable :value="codeSnippetList" :rowClass="rowClass" responsiveLayout="scroll">
      <Column field="id" header="id"></Column>
      <Column field="content" header="Content"></Column>
      <Column field="description" header="Description"></Column>
    </DataTable>
  </div>
</template>

<style scoped lang="scss">
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

::v-deep(.row-selected) {
  background-color: rgb(156, 146, 1) !important;
}
</style>
