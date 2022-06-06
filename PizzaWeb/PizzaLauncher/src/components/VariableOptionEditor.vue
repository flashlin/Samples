<template>
  <div>
    <!-- <PreviewFrame
      v-on:content="state.previewContent"
      style="{`with:600px; height:300px;`}
    /> -->

    <BlockUI :blocked="state.isBlocked">
      <DataTable
        :value="state.modelValue"
        editMode="cell"
        class="editable-cells-table"
        @cell-edit-complete="handleCellEditComplete"
        v-model:expandedRows="state.expandedRows"
        :lazy="true"
        :rowHover="true"
        responsiveLayout="scroll"
      >
        <template #header>
          <div class="flex justify-content-center align-items-center">
            <span>
              <Button icon="pi pi-refresh" @click="reloadAsync"/>
            </span>
          </div>
        </template>
        <Column :expander="true" headerStyle="width: 1rem" />
        <Column field="varName" header="Variable Name"></Column>
        <Column field="varType" header="Variable Type"></Column>
        <Column field="resxName" header="Resx Key Name">
          <template #editor="slotProps">
            <ComboSelect v-model="slotProps.data.resxName" 
              :options="() => getResxNamesAsync(slotProps.data.varType)" />
          </template>
        </Column>
        <Column header="Actions">
          <template #body="slotProps">
            <Button
              icon="pi pi-plus-circle"
              @click="handleAddResx()"
            />
            &nbsp;
            <Button
              icon="pi pi-save"
              @click="handleApplyVariableSetting(slotProps)"
            />
          </template>
        </Column>
        <template #expansion="slotProps">
          <div class="orders-subtable">
            <ResxTable 
              :resxName="slotProps.data.resxName"
              :varType="slotProps.data.varType"
              v-model="slotProps.data.resxList" />
          </div>
        </template>
      </DataTable>
    </BlockUI>
  </div>
</template>

<script setup lang="ts">
import { onMounted, PropType, reactive, ref } from "vue";
import {
  BannerApi,
  GetBannerReq,
  ITemplateData,
  IBannerTemplateEntity,
  ITemplateVariable,
IBannerSetting,
IAddBanner,
IVariableOption,
IBannerVariable,
IVariableResx,
IGetResxDataReq,
} from "@/models/Api";
//import PreviewFrame from "@/components/PreviewFrame";
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
import Editor from "@/components/Editor.vue";
import ResxTable from "@/components/ResxTable.vue";
import { IOption } from "@/typings/ui-typeings";
import ComboSelect from "@/components/ComboSelect.vue";

interface IBannerVariableSetting {
  varName: string;
  varType: string;
  resxName: string;
  resxList: IVariableResx[];
}

const props = defineProps({
  modelValue: {
    type: Object as PropType<IBannerVariableSetting[]>,
    default: () => ([] as IBannerVariable[]),
  },
  bannerName: {
    type: String as PropType<string>,
    required: true,
    default: "",
  }
});

const state = reactive({
  isEdit: false,
  isBlocked: false,
  modelValue: props.modelValue,
  expandedRows: [] as IVariableResx[],
  bannerIdSelected: "",
  previewContent: "",
});

const api = new BannerApi();

const reloadAsync = async () => {
  state.isBlocked = true;
  state.isBlocked = false;
}

async function getResxNamesAsync(varType: string) {
  let resx = await api.getResxNamesAsync(varType);
  return resx.map(x => ({
     label: x.resxName,
     value: x.resxName,
  } as IOption));
}

function handleCellEditComplete(event: DataTableCellEditCompleteEvent) {
  let { data, newValue, field } = event;
  data[field] = newValue;
}

function handleAddResx() {
  
}

function handleSearchVarType(event: AutoCompleteCompleteEvent){
  // setTimeout(() => {
  //   if (!event.query.trim().length) {
  //     state.filteredVarTypes = [...state.templateVariableOptions];
  //   }
  //   else {
  //     state.filteredVarTypes = state.templateVariableOptions.filter((item) => {
  //       return item.label.toLowerCase().startsWith(event.query.toLowerCase());
  //     });
  //   }
  // }, 250);
}

async function handleApplyVariableSetting(slotProps: ColumnRowSlots) {
  const variableSetting = state.modelValue[slotProps.index];
  await api.updateBannerVariableOptionAsync({
    bannerName: props.bannerName,
    varName: variableSetting.varName,
    varType: variableSetting.varType,
    resxName: variableSetting.resxName,
  });
  toastInfo(`Banner '${props.bannerName}' ${variableSetting.varName} updated`);
  // reloadAsync();
}

async function handleRestoreVariableSetting(slotProps: ColumnRowSlots) {
  // let resp = await confirmPopupAsync(
  //   `Are you sure you want to delete this '${slotProps.data.templateName}' template?`
  // );
  // if (resp) {
  //   let templateName = slotProps.data.templateName;
  //   await api.deleteTemplateAsync(templateName);
  //   toastInfo(`Delete ${templateName} Template Success`);
  //   await reloadAsync();
  //   return;
  // }
  // let templateName = slotProps.data.templateName;
  // toastInfo(`Cancel Delete ${templateName} Template`);
}

function onRowExpand(event: DataTableRowExpandEvent) {
  let data = event.data as any as ITemplateData;
  //state.expandedRows = data.variables;
  console.log("rowExpand", data);
}

function onRowCollapse(event: DataTableRowCollapseEvent) {
  //state.expandedRows = [];
}

onMounted(() => {
  reloadAsync();
});

// const onClickPreview = async () => {
//   let bannerTemplateName = state.bannerTemplateCheckedList.find(
//     (x) => typeof x == "string"
//   ) as string;
//   if (bannerTemplateName == null) {
//     message.info("Please select a template");
//     return;
//   }

//   message.info(`preview bannerTemplateId`);

//   let apiUrl = "http://localhost:5129";
//   let data = JSON.stringify(
//     new GetBannerReq({
//       bannerName: bannerTemplateName,
//     })
//   );

//   let content = await api.getBannerAsync(
//     new GetBannerReq({
//       bannerName: bannerTemplateName,
//     })
//   );
//   state.previewContent = content;
// };
</script>

<style lang="scss" scoped>
</style>