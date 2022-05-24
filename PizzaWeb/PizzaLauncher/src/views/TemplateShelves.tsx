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

import "./TemplateShelves.scss";

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

export default defineComponent({
  props: {},
  setup(props, { slots }) {
    const message = useMessage();

    const createBannerTemplateDataTableColumns =
      (): DataTableColumns<IBannerTemplateData> => {
        return [
          { type: "selection", options: ["all", "none"] },
          {
            title: "Name",
            key: "templateName",
            width: 200,
          },
          {
            type: "expand",
            renderExpand: (rowData: IBannerTemplateData) => {
              state.expansionColumns = createTemplateVariableDataTableColumns();
              return (
                <div>
                  <NLayout has-sider>
                    <NLayoutSider content-style="padding: 24px;">
                      Template Variables
                    </NLayoutSider>
                    <NLayoutContent content-style="padding: 24px; padding-left: 90%">
                      <NButton
                        strong={true}
                        tertiary={true}
                        size="small"
                        onClick={() => handleAddTemplateVariable(rowData)}
                      >
                        Add
                      </NButton>
                    </NLayoutContent>
                  </NLayout>
                  <NDataTable
                    rowKey={(rowData) => rowData.name}
                    data={rowData.variables}
                    columns={state.expansionColumns}
                  ></NDataTable>
                </div>
              );
            },
            width: 32,
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
      };

    const createTemplateVariableDataTableColumns =
      (): DataTableColumns<ITemplateVariable> => {
        return [
          { type: "selection", options: ["all", "none"] },
          {
            title: "Name",
            key: "name",
            width: 200,
            render(row: ITemplateVariable) {
              return (
                <div>
                  <ShowOrEdit value={row.name} onUpdateValue={(val) => row.name = val} />
                </div>
              );
            },
          },
          {
            title: "Type",
            key: "variableType",
            render(row: ITemplateVariable) {
              return (
                <div>
                  <NSelect
                    value={row.fulltype}
                    options={state.templateVariableOptions}
                    onUpdate:value={(value, option) => handleSelectVariableType(row, value, option)}
                    filterable
                    tag
                  />
                </div>
              );
            },
          },
          {
            title: "Action",
            key: "actions",
            render(row: ITemplateVariable) {
              return (
                <div>
                  <NButton
                    strong={true}
                    tertiary={true}
                    size="small"
                    onClick={() => handleDeleteTemplateVariable(row)}
                  >
                    Delete
                  </NButton>
                </div>
              );
            },
          },
        ];
      };

    const state = reactive({
      columns: createBannerTemplateDataTableColumns(),
      expansionColumns: [] as DataTableColumns<ITemplateVariable>,
      isEdit: false,
      templateList: [] as IBannerTemplateData[],
      bannerTemplateCheckedList: [] as boolean[],
      templateVariableOptions: [
        { label: "String", value: "String", disabled: false },
        { label: "Url(production)", value: "Url(production)", disabled: false },
        { label: "Image(200,100)", value: "Image(200,100)" },
      ],
      editingRow: null as unknown as IBannerTemplateData,
      activeKey: "",
      expandedRows: [],
      bannerIdSelected: "",
      currentEditId: "",
      previewContent: "abc",
    });

    const test = (v: any, row: any) => 
    {
      console.log("test", v, row);
    };

    const handleVariableNameOnChanged = (newValue: string, row: ITemplateVariable) => {
      row.name = newValue;
    };

    const handleAddTemplateVariable = (row: IBannerTemplateData) => {
      state.editingRow = row;
      row.variables.push({
        name: "",
        fulltype: "String",
      });
    };

    const handleSelectVariableType = (row: ITemplateVariable, value: string, option: SelectOption) => {
      row.fulltype = value;
      message.info('select: ' + JSON.stringify(row));
    };

    const handleApplyBannerTemplate = async (row: IBannerTemplateData) => {
      await api.updateTemplateAsync(row);
      message.info(`Update ${row.templateName} template success`);
    };

    const handleDeleteTemplateVariable = (row: ITemplateVariable) => {
      const index = state.editingRow.variables.indexOf(row);
      state.editingRow.variables.splice(index, 1);
    };

    const api = new BannerApi();

    const editor = ref<IEditorProxy>();

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
      console.log("selected", state.bannerIdSelected);
      let content = await api.getBannerAsync(
        new GetBannerReq({
          bannerId: state.bannerIdSelected,
        })
      );
      console.log("preview", content);
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
          rowKey={(rowData) => rowData.id}
          v-model:checked-row-keys={state.bannerTemplateCheckedList}
        ></NDataTable>
      </div>
    );
  },
});
