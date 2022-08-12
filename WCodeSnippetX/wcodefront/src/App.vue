<script setup lang="ts">
import { reactive, ref } from 'vue';
import { CodeSnippet, useCodeSnippetService } from './models';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
//import ColumnGroup from 'primevue/columngroup'; 
//import Row from 'primevue/row';

//let port = await window.__backend.getPort();

let codeSnippetList = reactive<CodeSnippet[]>([]);

async function loadData() {
  let app = useCodeSnippetService();
  let data: CodeSnippet[] = await app.queryAsync('');
  for(let item of data) {
    codeSnippetList.push(item);
  }
}

loadData();

</script>

<template>
  <DataTable :value="codeSnippetList" 
    hoverable="true"
    responsiveLayout="scroll">
    <Column field="id" header="id"></Column>
    <Column field="content" header="Content"></Column>
    <Column field="description" header="Description"></Column>
  </DataTable>
</template>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}


.ui-datatable tr.ui-datatable-even:hover, .ui-datatable tr.ui-datatable-odd:hover {
    background: #78BCFF;
}

</style>
