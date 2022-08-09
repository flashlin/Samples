<script setup lang="ts">
// This starter template is using Vue 3 <script setup> SFCs
// Check out https://vuejs.org/api/sfc-script-setup.html#script-setup
import { reactive } from 'vue';
import { CodeSnippet, useCodeSnippetService } from './models';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import ColumnGroup from 'primevue/columngroup'; 
import Row from 'primevue/row';

await CefSharp.BindObjectAsync("__backend");
let port = await window.__backend.getPort();
alert("port" + port);


let codeSnippetList = reactive<CodeSnippet[]>([]);
let app = useCodeSnippetService();
codeSnippetList = await app.queryAsync('');

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
