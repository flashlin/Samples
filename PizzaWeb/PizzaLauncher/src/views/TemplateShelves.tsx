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
} from "naive-ui";
import ShowOrEdit from "@/components/ShowOrEdit";
import ShowOrTextArea from "@/components/ShowOrTextArea";

export default defineComponent({
  props: {},
  setup(props, { slots }) {
    const templateVariableMenuOptions: MenuOption[] = [
      {
        label: () => <div>Hear the Wind Sing</div>,
        key: "hear-the-wind-sing",
      },
    ];

    const createBannerTemplateDataTableColumns =
      (): DataTableColumns<IBannerTemplateData> => {
        return [
          { type: "selection", options: ["all", "none"] },
          { title: "Name", key: "templateName", width: 100 },
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
                      <NButton strong={true} tertiary={true} size="small">
                        Add
                      </NButton>
                    </NLayoutContent>
                  </NLayout>
                  <NDataTable
                    rowKey={(rowData) => rowData.name}
                    data={rowData.variables}
                    v-model:columns={state.expansionColumns}
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
                  onClick: () => updateBannerTemplate(row),
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
          { title: "Name", key: "name", width: 100 },
          { title: "Type", key: "variableType" },
          {
            title: "Action",
            key: "actions",
            render(row: ITemplateVariable) {
              return h(
                NButton,
                {
                  strong: true,
                  tertiary: true,
                  size: "small",
                  onClick: () => applyTemplateVariable(row),
                },
                { default: () => "Apply" }
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
      activeKey: "",
      expandedRows: [],
      bannerIdSelected: "",
      currentEditId: "",
      previewContent: "abc",
    });

    const updateBannerTemplate = async (row: IBannerTemplateData) => {
      await api.updateTemplateAsync(row);
    };

    const applyTemplateVariable = (row: ITemplateVariable) => {};

    const api = new BannerApi();

    const isEditing = () => {
      return state.currentEditId != "";
    };

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

    const isEditingItem = (id: string) => {
      return state.currentEditId == id;
    };

    const onClickReload = async () => {
      let api = new BannerApi();
      let resp = await api.getAllTemplatesAsync();
      state.templateList = resp;
    };

    const onClickEditItem = (id: string) => {
      state.currentEditId = id;
    };

    const onClickUpdateContent = async (id: string) => {
      let item = state.templateList.find((x) => x.id == id)!;
      let idx = state.templateList.indexOf(item);
      //let newContent = editorRefs.value[idx].getContent();
      let newContent = editor.value!.getContent();

      let newItem = Object.assign({} as IBannerTemplateEntity, item);
      newItem.templateContent = newContent;

      await api.updateTemplateAsync(newItem);

      item.templateContent = newContent;
      state.currentEditId = "";
    };

    const onClickCancelContent = (id: string) => {
      let item = state.templateList.find((x) => x.id == id)!;
      let idx = state.templateList.indexOf(item);
      state.currentEditId = "";
    };

    const onClickSelectBannerId = (idx: number, id: string) => {
      for (let i = 0; i < state.bannerTemplateCheckedList.length; i++) {
        if (i != idx) {
          state.bannerTemplateCheckedList[i] = false;
        }
      }
      if (state.bannerTemplateCheckedList[idx]) {
        state.bannerIdSelected = id;
      } else {
        state.bannerIdSelected = "";
      }
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
