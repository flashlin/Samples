<script setup lang="ts">
import {reactive} from 'vue';
import {CodeSnippet, useCodeSnippetService, type IAppState} from './models';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
//import ColumnGroup from 'primevue/columngroup';
//import Row from 'primevue/row';

const data = reactive<IAppState>({
  selectedIndex: -1,
  selectedItem: CodeSnippet.Empty,
  searchText: '',
  codeSnippetList: [],
  filterCodes: [],
});

async function loadData() {
  let app = useCodeSnippetService();
  let list: CodeSnippet[] = await app.queryAsync(data.searchText);
  for (let item of list) {
    data.codeSnippetList.push(item);
  }
  if (list.length > 0) {
    data.selectedIndex = 0;
    data.selectedItem = list[0];
  }
}

function rowClass(item: CodeSnippet) {
  let css = '';
  if (item.id == data.selectedItem.id) {
    css = 'row-selected';
  }
  return css;
}

function handleKeyDown(event: KeyboardEvent) {
  if (event.key == 'ArrowDown' && data.selectedIndex < data.codeSnippetList.length - 1) {
    data.selectedItem = data.codeSnippetList[data.selectedIndex + 1];
    data.selectedIndex++;
    return;
  }

  if (event.key == 'ArrowUp' && data.selectedIndex > 0) {
    data.selectedItem = data.codeSnippetList[data.selectedIndex - 1];
    data.selectedIndex--;
    return;
  }

  if(event.key == 'Escape') {
    return;
  }

  if(event.key == 'Enter') {
    return;
  }
}

document.addEventListener('keydown', handleKeyDown);

loadData();
</script>

<template>
  <DataTable :value="data.codeSnippetList" :row-class="rowClass" responsive-layout="scroll">
    <Column field="id" header="id"></Column>
    <Column field="content" header="Content"></Column>
    <Column field="description" header="Description"></Column>
  </DataTable>
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
