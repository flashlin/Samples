<template>
  <div>
    <BlockUI :blocked="state.isBlocked">
      <DataTable :value="state.modelValue" editMode="cell" key="isoLangCode" class="editable-cells-table"
        @cell-edit-complete="handleCellEditComplete" :lazy="true" :rowHover="true" responsiveLayout="scroll">
        <template #header>
          <div class="flex justify-content-center align-items-center">
            <h5 class="m-0">All Contents for ResxName:'{{ props.resxName }}' VarType:'{{ props.varType }}'</h5>
            <span>
              <Button icon="pi pi-refresh" @click="reloadAsync" />
              <Button icon="pi pi-plus" @click="handleAddResx" />
              <Button icon="pi pi-save" @click="handleSaveResxAsync" />
            </span>
          </div>
        </template>
        <Column field="isoLangCode" header="IsoLangCode">
          <template #editor="slotProps">
            <ComboSelect v-model="slotProps.data.isoLangCode"
              :options="() => getAvailableIsoLangCodesOptionsAsync(slotProps.data)" />
          </template>
        </Column>
        <Column field="content" header="Content">
          <template #editor="slotProps">
            <InputText v-model="slotProps.data.content" autofocus />
          </template>
        </Column>
        <Column header="Actions">
          <template #body="slotProps">
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
import ComboSelect from "@/components/ComboSelect.vue";
import { IOption } from "@/typings/ui-typeings";
import { BannerApi } from "@/models/Api";

const api = new BannerApi();

const props = defineProps({
  resxName: {
    type: String,
    required: true,
  },
  varType: {
    type: String,
    required: true,
  },
  modelValue: {
    type: Object as PropType<IVariableResx[]>,
    default: () => ([]),
  },
});

const state = reactive({
  isEdit: false,
  isBlocked: false,
  modelValue: props.modelValue,
});

const reloadAsync = async () => {
  state.isBlocked = true;
  state.isBlocked = false;
}

function handleCellEditComplete(event: DataTableCellEditCompleteEvent) {
  let { data, newValue, field } = event;
  data[field] = newValue;
}

async function handleSaveResxAsync() {
  await api.upsertResxAsync({
    resxName: props.resxName,
    varType: props.varType,
    contentList: state.modelValue,
  });
  toastInfo(`${props.resxName} '${props.varType}' Saved`);
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
  let isoLangCodes = getAvailableIsoLangCodes();
  if (isoLangCodes.length == 0) {
    return;
  }
  state.modelValue.push({
    isoLangCode: isoLangCodes[0],
    content: "",
  });
}

function getAvailableIsoLangCodesOptionsAsync(resx?: IVariableResx): Promise<IOption[]> {
  return new Promise((resolve)=>{
    let isoLangCodes = getAvailableIsoLangCodes(resx);
    resolve(isoLangCodes.map(x=>({
      label: x,
      value: x
    } as IOption)));
  });
}

function getAvailableIsoLangCodes(resx?: IVariableResx) {
  let currentAllIsoLangCodes = state.modelValue.map(item => {
    return item.isoLangCode;
  });
  return AllIsoLangCodes.filter(isoLangCode => {
    if (resx != null && resx.isoLangCode == isoLangCode) {
      return true;
    }
    return !currentAllIsoLangCodes.includes(isoLangCode);
  });
}

onMounted(() => {
  reloadAsync();
});
</script>

<style lang="scss" scoped>
</style>