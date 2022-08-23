<script setup lang="ts">
import { reactive } from 'vue';
import { CodeSnippet, useCodeSnippetService, type IAppState } from './models';
import DataTable, { DataTableRowSelectEvent } from 'primevue/datatable';
import Column from 'primevue/column';
//import ColumnGroup from 'primevue/columngroup';
//import Row from 'primevue/row';
import InputText from 'primevue/inputtext';
import Button from 'primevue/button';
import ConfirmDialog from 'primevue/confirmdialog';
import DynamicDialog from 'primevue/dynamicdialog';
import { useDialog } from 'primevue/usedialog';
import { useConfirm } from "primevue/useconfirm";
import Toast from 'primevue/toast';
import { useToast } from "primevue/usetoast";
import AddCodeSnippet from './views/AddCodeSnippet.vue';

const data = reactive<IAppState>({
  selectedIndex: -1,
  selectedItem: CodeSnippet.Empty,
  searchText: '',
  codeSnippetList: [],
  isEditingData: false,
});

const codeSnippetService = useCodeSnippetService();
const dialog = useDialog();
const confirm = useConfirm();
const toast = useToast();

async function queryData() {
  let list: CodeSnippet[] = await codeSnippetService.queryCodeAsync(data.searchText);
  data.codeSnippetList = [];
  for (let item of list) {
    data.codeSnippetList.push(item);
  }
  data.selectedIndex = -1;
  if (list.length > 0 && (data.selectedIndex >= list.length || data.selectedIndex == -1)) {
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

function onClickEdit() {
  if (data.isEditingData) {
    return;
  }
  data.isEditingData = true;
  dialog.open(AddCodeSnippet, {
    props: {
      header: 'Edit Code Snippet',
      modal: true,
    },
    data: data.selectedItem,
    onClose: (options) => {
      data.isEditingData = false;
      const editingData = options!.data;
      if (editingData == undefined) {
        info('No data to update');
        return;
      }
      queryData();
      info('update success');
    },
  });
}


function onClickAdd() {
  if (data.isEditingData) {
    return;
  }
  data.isEditingData = true;
  dialog.open(AddCodeSnippet, {
    props: {
      header: 'Add Code Snippet',
      modal: true,
    },
    data: new CodeSnippet(),
    onClose: (options) => {
      data.isEditingData = false;
      const editingData = options!.data;
      if (editingData == undefined) {
        info('No data to add');
        return;
      }
      queryData();
      info('add success');
    },
  });
}

function onClickDelete() {
  if( data.selectedIndex == -1) {
    return;
  }
  confirm.require({
    message: 'Are you sure you want to delete?',
    header: 'Confirmation',
    icon: 'pi pi-exclamation-triangle',
    accept: async () => {
      await codeSnippetService.deleteCodeAsync(data.selectedItem.id);
      info('Delete Success');
      data.selectedIndex = -1;
      queryData();
    },
    reject: () => {
      info('Delete canceled');
    }
  });
}

function onRowSelect(event: DataTableRowSelectEvent) {
  data.selectedIndex = event.index;
  data.selectedItem = data.codeSnippetList[event.index];
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
    codeSnippetService.minimizeAsync();
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
  <ConfirmDialog />
  <Button label="Update" class="p-button p-button-info" icon="pi pi-pencil" iconPos="left" @click="onClickEdit" />
  <Button label="Add" icon="pi pi-plus" iconPos="left" class="p-button p-button-info" @click="onClickAdd" />
  <Button label="Delete" icon="pi pi-times" iconPos="left" class="p-button p-button-danger" @click="onClickDelete" />
  <DataTable :value="data.codeSnippetList" :row-class="rowClass" selectionMode="single" @rowSelect="onRowSelect"
    responsive-layout="scroll">
    <Column field="id" header="id"></Column>
    <Column field="programLanguage" header="Language"></Column>
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

:deep(.row-selected) {
  background-color: rgb(156, 146, 1) !important;
}

:deep(tr:not(.p-highlight):hover) {
  color: rgb(247, 230, 0) !important;
  background-color: rgb(87, 56, 56) !important;
}
</style>
