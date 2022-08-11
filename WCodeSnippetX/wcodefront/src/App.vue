<script setup lang="ts">
import { reactive, ref } from 'vue';
import { CodeSnippet, useCodeSnippetService } from './models';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
//import ColumnGroup from 'primevue/columngroup'; 
//import Row from 'primevue/row';

//let port = await window.__backend.getPort();


let codeSnippetList = reactive<CodeSnippet[]>([]);
//let app = useCodeSnippetService();
//codeSnippetList = await app.queryAsync('');

async function loadData() {
  await CefSharp.BindObjectAsync("__backend");
  let resp = await window.__backend.queryCode('');
  let list: CodeSnippet[] = JSON.parse(resp);
  for(let item of list) {
    codeSnippetList.push(item);
    codeSnippetList.push({
      content: "test",
      programLanguage: "vue",
      description: "test",
      id: 123
    });
    //alert(JSON.stringify(item));
  }
}

loadData();

</script>

<template>
  <DataTable :value="codeSnippetList" responsiveLayout="scroll">
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
</style>
