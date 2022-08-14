<script setup lang="ts">
import { reactive } from 'vue';
import { CodeSnippet, useCodeSnippetService, type IAppState } from './models';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
//import ColumnGroup from 'primevue/columngroup';
//import Row from 'primevue/row';
import InputText from 'primevue/inputtext';
import Button from 'primevue/button';
import DynamicDialog from 'primevue/dynamicdialog';
import { useDialog } from 'primevue/usedialog';
import Toast from 'primevue/toast';
import { useToast } from "primevue/usetoast";
import AddCodeSnippet from './views/AddCodeSnippet.vue';

const data = reactive<IAppState>({
  selectedIndex: -1,
  selectedItem: CodeSnippet.Empty,
  searchText: '',
  codeSnippetList: [],
  filterCodes: [],
  isEditingData: false,
});

const codeSnippetService = useCodeSnippetService();
const dialog = useDialog();
const toast = useToast();

async function queryData() {
  let list: CodeSnippet[] = await codeSnippetService.queryCodeAsync(data.searchText);
  data.codeSnippetList = [];
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

function onSearchChanged() {
  queryData();
}

function onSearchEnter() {
  codeSnippetService.setClipboardAsync(data.selectedItem.content);
}

function info(message: string) {
  toast.add({
    severity: 'info',
    summary: 'Info Message',
    detail: message,
    life: 3000
  });
}

function onClickAdd() {
  if (data.isEditingData) {
    return;
  }
  data.isEditingData = true;
  dialog.open(AddCodeSnippet, {
    props: {
      header: 'Add/Edit Code Snippet',
      modal: true,
    },
    data: data.selectedItem,
    onClose: (options) => {
      data.isEditingData = false;
      const editingData = options!.data;
      if (editingData == undefined) {
        info('No data to add/edit');
        return;
      }
      if (editingData.id == 0) {
        info('add success');
        return;
      }
      info('update success');
    },
  });
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

  if (event.key == 'Escape') {
    return;
  }

  if (event.key == 'Enter') {
    codeSnippetService.setClipboardAsync(data.selectedItem.content);
    return;
  }
}

document.addEventListener('keydown', handleKeyDown);

queryData();
</script>

<template>
  <Toast />
  <DynamicDialog />
  <Button label="Add" class="p-button p-component p-button-icon-only p-button-rounded" @click="onClickAdd">
    <span class="pi pi-plus p-button-icon"></span>
  </Button>
  <DataTable :value="data.codeSnippetList" :row-class="rowClass" responsive-layout="scroll">
    <Column field="id" header="id"></Column>
    <Column field="content" header="Content"></Column>
    <Column field="description" header="Description"></Column>
  </DataTable>
  <span class="p-input-icon-left" style="width: 100%;">
    <i class="pi pi-search" />
    <InputText type="text" v-model="data.searchText" style="width: 100%;" @input="onSearchChanged()"
      @keydown.enter="onSearchEnter()" placeholder="Search" />
  </span>
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
