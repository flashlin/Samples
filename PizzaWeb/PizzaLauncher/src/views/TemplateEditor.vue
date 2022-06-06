<template>
  <div>
    <!-- <PreviewFrame
      v-on:content="state.previewContent"
      style="{`with:600px; height:300px;`}
    /> -->

    <BlockUI :blocked="state.isBlocked">
      <DataTable :value="state.templateList" dataKey="templateName" editMode="cell" class="editable-cells-table"
        @cell-edit-complete="handleCellEditComplete" 
        v-model:expandedRows="state.expandedRows" 
        @rowExpand="onRowExpand"
        @rowCollapse="onRowCollapse" :lazy="true" :rowHover="true" responsiveLayout="scroll">
        <template #header>
          <div class="flex justify-content-center align-items-center">
            <h5 class="m-0">Templates</h5>
            <span>
              <Button icon="pi pi-plus" @click="handleAddTemplate" />
              &nbsp;
              <Button icon="pi pi-refresh" @click="reloadAsync" />
            </span>
          </div>
        </template>
        <Column :expander="true" headerStyle="width: 1rem" />
        <Column header="Selected"></Column>
        <Column field="templateName" header="Name">
          <template #editor="slotProps">
            <InputText v-model="slotProps.data.templateName" autofocus />
          </template>
        </Column>
        <Column field="templateContent" header="Content">
          <template #body="slotProps">
            {{ subContent(slotProps.data.templateContent) }}
          </template>
          <template #editor="slotProps">
            <Editor v-model="slotProps.data.templateContent" editorStyle="height: 320px" />
          </template>
        </Column>
        <Column header="Actions">
          <template #body="slotProps">
            <Button icon="pi pi-save" @click="handleApplyAddTemplate(slotProps)" />
            &nbsp;
            <Button icon="pi pi-trash" @click="handleDeleteTemplate(slotProps)" />
          </template>
        </Column>
        <template #footer> In {{ state.indexPage }} Index. </template>
        <template #expansion="slotProps">
          <div class="orders-subtable">
            <h5>Orders for {{ slotProps.data.templateName }}</h5>
            <DataTable :value="slotProps.data.variables" dataKey="varName" editMode="cell"
              @cell-edit-complete="handleCellEditComplete" class="editable-cells-table" responsiveLayout="scroll">
              <template #header>
                <div class="flex justify-content-center align-items-center">
                  <h5 class="m-0">Variables</h5>
                  <span>
                    <Button icon="pi pi-plus" @click="handleAddTemplateVariable(slotProps.data.variables)" />
                    &nbsp;
                    <Button icon="pi pi-refresh" />
                  </span>
                </div>
              </template>
              <Column field="varName" header="name" sortable>
                <template #editor="slotProps">
                  <InputText v-model="slotProps.data.varName" autofocus />
                </template>
              </Column>
              <Column field="varType" header="type">
                <template #editor="slotProps">
                  <VarTypeComboSelect v-model="slotProps.data.varType" />
                  <!-- <AutoComplete v-model="slotProps.data.varType" :suggestions="state.filteredVarTypes"
                    @complete="handleSearchVarType($event)" field="label" :dropdown="true" /> -->
                </template>
              </Column>
              <Column headerStyle="width:4rem">
                <template #body="varSlotProps">
                  <Button icon="pi pi-trash" @click="handleDeleteVariable(slotProps.data.variables, varSlotProps)" />
                </template>
              </Column>
            </DataTable>
          </div>
        </template>
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
//import Editor from 'primevue/editor';
import Editor from "@/components/Editor.vue";
import { DefaultTemplateVariableOptions, IOption } from "@/typings/ui-typeings";
import VarTypeComboSelect from "@/components/VarTypeComboSelect.vue";

const state = reactive({
  isEdit: false,
  indexPage: 0,
  isBlocked: false,
  templateList: [] as ITemplateData[],
  templateVariableOptions: DefaultTemplateVariableOptions,
  expandedRows: [] as ITemplateVariable[],
  filteredVarTypes: [] as IOption[],
  bannerIdSelected: "",
  previewContent: "",
});

const api = new BannerApi();

const reloadAsync = async () => {
  state.isBlocked = true;
  let resp = await api.getAllTemplatesAsync(state.indexPage);
  state.templateList = resp;
  state.isBlocked = false;
}

function handleCellEditComplete(event: DataTableCellEditCompleteEvent) {
  let { data, newValue, field } = event;
  data[field] = newValue;
}

function handleSearchVarType(event: AutoCompleteCompleteEvent) {
  setTimeout(() => {
    if (!event.query.trim().length) {
      state.filteredVarTypes = [...state.templateVariableOptions];
    }
    else {
      state.filteredVarTypes = state.templateVariableOptions.filter((item) => {
        return item.label.toLowerCase().startsWith(event.query.toLowerCase());
      });
    }
  }, 250);
}

function handleAddTemplate() {
  state.templateList.push({
    id: 0,
    templateName: "unknown",
    templateContent: "<div></div>",
    variables: [],
  });
}

async function handleApplyAddTemplate(slotProps: ColumnRowSlots) {
  const template = state.templateList[slotProps.index];
  if (template.id === 0) {
    await api.addTemplateAsync(template);
    toastInfo(`Template '${template.templateName}' added`);
  } else {
    await api.updateTemplateAsync(template);
    toastInfo(`Template '${template.templateName}' updated`);
  }
  reloadAsync();
}

async function handleDeleteTemplate(slotProps: ColumnRowSlots) {
  let resp = await confirmPopupAsync(
    `Are you sure you want to delete this '${slotProps.data.templateName}' template?`
  );
  if (resp) {
    let templateName = slotProps.data.templateName;
    await api.deleteTemplateAsync(templateName);
    let resp = await api.getAllTemplatesAsync(state.indexPage);
    state.templateList = resp;
    toastInfo(`Delete ${templateName} Template Success`);
    return;
  }
  let templateName = slotProps.data.templateName;
  toastInfo(`Cancel Delete ${templateName} Template`);
}

function onRowExpand(event: DataTableRowExpandEvent) {
  let data = event.data as any as ITemplateData;
  //state.expandedRows = data.variables;
  console.log("rowExpand", data);
}

function onRowCollapse(event: DataTableRowCollapseEvent) {
  //state.expandedRows = [];
}

function handleAddTemplateVariable(vars: ITemplateVariable[]) {
  vars.push({
    varName: "unknown",
    varType: "String",
  });
}

function handleDeleteVariable(data: any, slotProps: ColumnRowSlots) {
  data.splice(slotProps.index, 1);
}

// const editor = ref<IEditorProxy>();

const subContent = (content: string): string => {
  if (content == null) {
    return "";
  }
  const maxLength = 100;
  if (content.length <= maxLength) {
    return content;
  }
  return content.substring(0, maxLength) + "...";
};

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
@import "./TemplateEditor.scss";
</style>