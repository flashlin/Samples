<template>
  <div>
    <Toast />
    <!-- <PreviewFrame
      v-on:content="state.previewContent"
      style="{`with:600px; height:300px;`}
    /> -->

    <DataTable
      :value="state.templateList"
      :lazy="true"
      :rowHover="true"
      responsiveLayout="scroll"
    >
      <template #header>
        <div class="flex justify-content-center align-items-center">
          <h5 class="m-0">Templates</h5>
          <span>
            <Button icon="pi pi-plus" @click="handleAddTemplate" />
            &nbsp;
            <Button icon="pi pi-refresh" />
          </span>
        </div>
      </template>
      <Column header="Selected"></Column>
      <Column field="templateName" header="Name"></Column>
      <Column field="templateContent" header="Content">
        <template #body="slotProps">
          {{ slotProps.data.templateContent }}
        </template>
      </Column>
      <Column header="Actions">
        <template #body>
          <Button icon="pi pi-save" />
          &nbsp;
          <Button icon="pi pi-trash" />
        </template>
      </Column>
      <template #footer> In {{ state.indexPage }} Index. </template>
    </DataTable>
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
//import Editor, { IEditorProxy } from "@/components/Editor";


import "primevue/resources/primevue.min.css";
import "primeicons/primeicons.css";
import "primevue/resources/themes/bootstrap4-dark-blue/theme.css";

import Button from "primevue/button";
import DataTable from "primevue/datatable";
import Column from "primevue/column";
import { useToast } from "primevue/usetoast";

const data = ref([]);

const state = reactive({
  isEdit: false,
  indexPage: 0,
  templateList: [] as ITemplateData[],
  templateVariableOptions: [
    { label: "String", value: "String", disabled: false },
    { label: "Url(production)", value: "Url(production)", disabled: false },
    { label: "Image(200,100)", value: "Image(200,100)" },
  ],
  editingRow: null as unknown as ITemplateData,
  expandedRows: [],
  bannerIdSelected: "",
  previewContent: "",
});

const toast = useToast();
function toastInfo(message: string) {
  toast.add({
    severity: "info",
    summary: "Info Message",
    detail: message,
    life: 3000,
  });
}

function handleAddTemplate() {
  state.templateList.push({
    id: 0,
    templateName: "templateName",
    templateContent: "<div></div>",
    variables: [],
  });
  toastInfo("Add empty template");
}

// const handleAddTemplateVariable = (row: IBannerTemplateData) => {
//   state.editingRow = row;
//   row.variables.push({
//     name: "",
//     fulltype: "String",
//   });
// };

// const handleSelectVariableType = (
//   row: ITemplateVariable,
//   value: string,
//   option: SelectOption
// ) => {
//   row.fulltype = value;
//   message.info("select: " + JSON.stringify(row));
// };

// const handleApplyBannerTemplate = async (row: IBannerTemplateData) => {
//   await api.updateTemplateAsync(row);
//   message.info(`Update ${row.templateName} template success`);
// };

// const handleDeleteTemplateVariable = (row: ITemplateVariable) => {
//   const index = state.editingRow.variables.indexOf(row);
//   state.editingRow.variables.splice(index, 1);
// };

const api = new BannerApi();

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

onMounted(async () => {
  let resp = await api.getAllTemplatesAsync(state.indexPage);
  //state.templateList = resp;
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