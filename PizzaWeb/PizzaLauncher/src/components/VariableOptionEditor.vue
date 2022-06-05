<template>
  <div>
    <!-- <PreviewFrame
      v-on:content="state.previewContent"
      style="{`with:600px; height:300px;`}
    /> -->

    <BlockUI :blocked="state.isBlocked">
      <DataTable
        :value="state.variablesOptions"
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
            <h5 class="m-0">All Variable Options</h5>
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
            <InputText v-model="slotProps.data.resxName" autofocus />
          </template>
        </Column>
        <Column header="Actions">
          <template #body="slotProps">
            <Button
              icon="pi pi-save"
              @click="handleApplyVariableSetting(slotProps)"
            />
            &nbsp;
            <Button
              icon="pi pi-trash"
              @click="handleRestoreVariableSetting(slotProps)"
            />
          </template>
        </Column>
      </DataTable>
    </BlockUI>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
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

interface IOption 
{
  label: string;
  value: string;
}

interface IBannerVariableSetting {
  varName: string;
  varType: string;
  resxName: string;
  resxList: IVariableResx[];
}

const props = defineProps({
  modelValue: {
    type: Object,
    default: () => ([] as IBannerVariable[]),
  },
});

const state = reactive({
  isEdit: false,
  isBlocked: false,
  variablesOptions: [] as IBannerVariableSetting[],
  expandedRows: [] as ITemplateVariable[],
  filteredVarTypes: [] as IOption[],
  bannerIdSelected: "",
  previewContent: "",
});

const api = new BannerApi();

const reloadAsync = async () => {
  state.isBlocked = true;
  state.isBlocked = false;
}

function handleCellEditComplete(event: DataTableCellEditCompleteEvent) {
  let { data, newValue, field } = event;
  data[field] = newValue;
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
  // const banner = state.bannerList[slotProps.index];
  // if (banner.id === 0) {
  //   await api.addBannerAsync(banner);
  //   toastInfo(`Banner '${banner.bannerName}' added`);
  // } else {
  //   await api.updateBannerAsync(banner);
  //   toastInfo(`Banner '${banner.templateName}' updated`);
  // }
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