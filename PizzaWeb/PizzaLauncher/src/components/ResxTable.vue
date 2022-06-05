<template>
  <div>
    <BlockUI :blocked="state.isBlocked">
      <DataTable :value="state.resx.resxList" editMode="cell" class="editable-cells-table"
        @cell-edit-complete="handleCellEditComplete" :lazy="true" :rowHover="true" responsiveLayout="scroll">
        <template #header>
          <div class="flex justify-content-center align-items-center">
            <h5 class="m-0">All Resx Options</h5>
            <span>
              <Button icon="pi pi-refresh" @click="reloadAsync" />
              <Button icon="pi pi-save" @click="handleAddResx" />
            </span>
          </div>
        </template>
        <Column :expander="true" headerStyle="width: 1rem" />
        <Column field="isoLangCode" header="IsoLangCode">
          <template #editor="slotProps">
            <ComboSelect v-model="slotProps.data.isoLangCode"
              :options="getIsoLangCode" />
          </template>
        </Column>
        <Column field="content" header="Content">
          <template #editor="slotProps">
            <InputText v-model="slotProps.data.content" autofocus />
          </template>
        </Column>
        <Column header="Actions">
          <template #body="slotProps">
            <Button icon="pi pi-save" @click="handleSaveResx(slotProps)" />
            &nbsp;
            <Button icon="pi pi-trash" @click="handleDeleteResx(slotProps)" />
          </template>
        </Column>
      </DataTable>
    </BlockUI>
  </div>
</template>

<script setup lang="ts">
import { onMounted, PropType, reactive, ref } from "vue";
import {
  GetBannerReq,
  ITemplateData,
  IBannerTemplateEntity,
  ITemplateVariable,
  IBannerSetting,
  IAddBanner,
  IVariableOption,
  IBannerVariable,
  IVariableResx,
AllIsoLangCodes,
} from "@/models/Api";
import BlockUI from "primevue/blockui";
import Button from "primevue/button";
import DataTable, {
  DataTableCellEditCompleteEvent,
  DataTableRowCollapseEvent,
  DataTableRowExpandEvent,
} from "primevue/datatable";
import Column, { ColumnSlots } from "primevue/column";
import InputText from "primevue/inputtext";
import AutoComplete, { AutoCompleteCompleteEvent } from "primevue/autocomplete";
import { confirmPopupAsync, toastInfo } from "@/models/AppToast";
import { ColumnRowSlots } from "@/typings/primevue-typings";

const props = defineProps({
  modelValue: {
    type: Object as PropType<IBannerVariable>,
    default: () => ([]),
  },
});

const state = reactive({
  isEdit: false,
  isBlocked: false,
  resx: props.modelValue,
});

const reloadAsync = async () => {
  state.isBlocked = true;
  state.isBlocked = false;
}

function handleCellEditComplete(event: DataTableCellEditCompleteEvent) {
  let { data, newValue, field } = event;
  data[field] = newValue;
}

function handleSaveResx(slotProps: ColumnRowSlots) {
  let { data } = slotProps;
}

function handleDeleteResx(slotProps: ColumnRowSlots) {
  let { data } = slotProps;
}


function onRowExpand(event: DataTableRowExpandEvent) {
  let data = event.data as any as ITemplateData;
  //state.expandedRows = data.variables;
  console.log("rowExpand", data);
}

function onRowCollapse(event: DataTableRowCollapseEvent) {
  //state.expandedRows = [];
}

function handleAddResx() {
  let isoLangCodes = getIsoLangCode();
}

function getIsoLangCode() {
  let currentAllIsoLangCodes = state.resx.resxList.map(item =>{
    return item.isoLangCode;
  });
  return AllIsoLangCodes.filter(item => {
    return !currentAllIsoLangCodes.includes(item);
  });
}

onMounted(() => {
  reloadAsync();
});
</script>

<style lang="scss" scoped>
</style>