import {
  computed,
  defineComponent,
  Fragment,
  onMounted,
  reactive,
  Ref,
  ref,
  h,
} from "vue";
import {
  BannerApi,
  GetBannerReq,
  IBannerTemplateData,
  IBannerTemplateEntity,
  ITemplateVariable,
} from "@/models/Api";
import PreviewFrame from "@/components/PreviewFrame";
import Editor, { IEditorProxy } from "@/components/Editor";

//import "./TemplateShelves.scss";

import {
  NButton,
  useMessage,
  DataTableColumns,
  NDataTable,
  NMenu,
  MenuOption,
  NLayout,
  NLayoutSider,
  NLayoutContent,
  NSelect,
  NInput,
  SelectOption,
} from "naive-ui";
import ShowOrEdit from "@/components/ShowOrEdit";
import ShowOrTextArea from "@/components/ShowOrTextArea";
import EditableSelect, {
  EditableSelectOption,
} from "@/components/EditableSelect";
import { CreateRowKey, RowKey } from "naive-ui/es/data-table/src/interface";

export default defineComponent({
  props: {},
  setup(props, { slots }) {
    const state = reactive({
      columns: null as null | DataTableColumns<IBannerTemplateData>,
      expansionColumns: [] as DataTableColumns<ITemplateVariable>,
      isEdit: false,
      templateList: [] as IBannerTemplateData[],
      bannerTemplateCheckedList: [] as RowKey[],
      templateVariableOptions: [
        { label: "String", value: "String", disabled: false },
        { label: "Url(production)", value: "Url(production)", disabled: false },
        { label: "Image(200,100)", value: "Image(200,100)" },
      ],
      editingRow: null as unknown as IBannerTemplateData,
      expandedRows: [],
      bannerIdSelected: "",
      previewContent: "",
    });
    const message = useMessage();

    state.columns = [
      { type: "selection", options: ["all", "none"] },
      {
        title: "Name",
        key: "templateName",
        width: 200,
      },
      {
        title: "Content",
        key: "templateContent",
        render(row, index) {
          return h(ShowOrTextArea, {
            value: row.templateContent,
            onUpdateValue(v: string) {
              state.templateList[index].templateContent = v;
            },
          });
        },
      },
      {
        title: "Action",
        key: "actions",
        render(row: any) {
          return h(
            NButton,
            {
              strong: true,
              tertiary: true,
              size: "small",
              onClick: () => handleApplyBannerTemplate(row),
            },
            { default: () => "Apply" }
          );
        },
      },
    ];


    const handleApplyBannerTemplate = async (row: IBannerTemplateData) => {
      await api.updateTemplateAsync(row);
      message.info(`Update ${row.templateName} template success`);
    };

    const handleDeleteTemplateVariable = (row: ITemplateVariable) => {
      const index = state.editingRow.variables.indexOf(row);
      state.editingRow.variables.splice(index, 1);
    };

    const api = new BannerApi();

    onMounted(async () => {
      let resp = await api.getAllTemplatesAsync();
      state.templateList = resp;
      state.bannerTemplateCheckedList = new Array(
        state.templateList.length
      ).fill(false);
    });

    const onClickReload = async () => {
      let api = new BannerApi();
      let resp = await api.getAllTemplatesAsync();
      state.templateList = resp;
    };

    const onClickPreview = async () => {
      let bannerTemplateName = state.bannerTemplateCheckedList.find(
        (x) => typeof x == "string"
      ) as string;
      if (bannerTemplateName == null) {
        message.info("Please select a template");
        return;
      }

      message.info(`preview bannerTemplateId`);

      let apiUrl = "http://localhost:5129";
      let data = JSON.stringify(
        new GetBannerReq({
          bannerName: bannerTemplateName,
        })
      );

      let content = await api.getBannerAsync(
        new GetBannerReq({
          bannerName: bannerTemplateName,
        })
      );
      state.previewContent = content;
    };

    /*

                            <Editor
                              content={slotProps.data.templateContent}
                              ref={editor}
                            />,
    */

    return () => (
      <div>
        <button onClick={onClickReload}>Reload</button>

        <button onClick={onClickPreview}>Preview</button>
        <PreviewFrame
          content={state.previewContent}
          style={`with:600px; height:300px;`}
        />

        <NDataTable
          data={state.templateList}
          v-model:columns={state.columns}
          rowKey={(row) => row.templateName}
          v-model:checkedRowKeys={state.bannerTemplateCheckedList}
        ></NDataTable>
      </div>
    );
  },
});
