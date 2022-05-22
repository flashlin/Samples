import {
  computed,
  defineComponent,
  Fragment,
  onMounted,
  reactive,
  Ref,
  ref,
} from "vue";
import { BannerApi, GetBannerReq, IBannerTemplateData, IBannerTemplateEntity } from "@/models/Api";
import PreviewFrame from "@/components/PreviewFrame";
import Editor, { IEditorProxy } from "@/components/Editor";

import "./TemplateShelves.scss";
import DataTable from "primevue/datatable";
import Button from "primevue/button";
import Column, { ColumnSlots } from "primevue/column";
import { ColumnRowSlots } from "@/typings/primevue-typings";

export default defineComponent({
  props: {},
  setup(props, { slots }) {
    const state = reactive({
      columns: [
        { field: "id", header: "id" },
        { field: "templateContent", header: "content" },
      ],
      templateList: [] as IBannerTemplateData[],
      bannerIdCheckedList: [] as boolean[],
      bannerIdSelected: "",
      currentEditId: "",
      previewContent: "abc",
    });

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
      state.bannerIdCheckedList = new Array(state.templateList.length).fill(
        false
      );
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
      for (let i = 0; i < state.bannerIdCheckedList.length; i++) {
        if (i != idx) {
          state.bannerIdCheckedList[i] = false;
        }
      }
      if (state.bannerIdCheckedList[idx]) {
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

    return () => (
      <div>
        <button onClick={onClickReload}>Reload</button>

        <button onClick={onClickPreview}>Preview</button>
        <PreviewFrame
          content={state.previewContent}
          style={`with:600px; height:300px;`}
        />

        <DataTable value={state.templateList} responsiveLayout="scroll">
          <slot name="header">
            <div class="table-header">
              Templates
              <Button icon="pi pi-refresh" />
            </div>
          </slot>
          <Column header="selected">
            {{
              body: (slotProps: ColumnRowSlots) => [
                <div>
                  <input
                    type="checkbox"
                    v-model={state.bannerIdCheckedList[slotProps.index]}
                    onChange={() =>
                      onClickSelectBannerId(slotProps.index, slotProps.data.templateName)
                    }
                  />
                </div>,
              ],
            }}
          </Column>
          <Column field="templateName" header="template name">
            {{
              body: (slotProps: ColumnRowSlots) => [
                <span>{slotProps.data.templateName}</span>,
              ],
            }}
          </Column>
          {!isEditing()
            ? [
                <Column field="templateContent" header="template">
                  {{
                    body: (slotProps: any) => [
                      subContent(slotProps.data.templateContent),
                    ],
                  }}
                </Column>,
                <Column header="operators">
                  {{
                    body: (slotProps: ColumnRowSlots) => [
                      <Button
                        onClick={() => onClickEditItem(slotProps.data.id)}
                      >
                        Edit
                      </Button>,
                    ],
                  }}
                </Column>,
              ]
            : [
                <Column field="templateContent" header="template">
                  {{
                    body: (slotProps: ColumnRowSlots) =>
                      !isEditingItem(slotProps.data.id)
                        ? []
                        : [
                            <Editor
                              content={slotProps.data.templateContent}
                              ref={editor}
                            />,
                          ],
                  }}
                </Column>,
                <Column header="operators">
                  {{
                    body: (slotProps: ColumnRowSlots) => [
                      <Button
                        onClick={() => onClickUpdateContent(slotProps.data.id)}
                      >
                        Update
                      </Button>,
                      <Button
                        onClick={() => onClickCancelContent(slotProps.data.id)}
                      >
                        Cancel
                      </Button>,
                    ],
                  }}
                </Column>,
              ]}
            {/* {{
              expansion: (slotProps: ColumnRowSlots) => [
                <div class="orders-subtable">
                    <h5>Variables for {slotProps.data.templateName}</h5>
                    <DataTable value={slotProps.data.variables} responsiveLayout="scroll">
                        <Column field="name" header="variable name" sortable></Column>
                        <Column field="variableType" header="type"></Column>
                    </DataTable>
                </div>
              ]
            }} */}
        </DataTable>
      </div>
    );
  },
});
