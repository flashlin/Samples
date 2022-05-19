import {
  computed,
  defineComponent,
  Fragment,
  onMounted,
  reactive,
  Ref,
  ref,
} from "vue";
import { BannerApi, IBannerTemplateEntity } from "@/models/Api";
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
      templateList: [] as IBannerTemplateEntity[],
      currentEditId: "",
      previewContent: "abc",
    });

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
      let api = new BannerApi();
      let resp = await api.getAllTemplatesAsync();
      state.templateList = resp;
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
      item.templateContent = newContent;
      //let resp = await BannerApi.updateBannerTemplate(item);
      state.currentEditId = "";
    };

    const onClickCancelContent = (id: string) => {
      let item = state.templateList.find((x) => x.id == id)!;
      let idx = state.templateList.indexOf(item);
      state.currentEditId = "";
    };

    return () => (
      <div>
        <button onClick={onClickReload}>Reload</button>

        <PreviewFrame
          content={state.previewContent}
          style={`with:100px; height:100px;`}
        />

        <DataTable value={state.templateList} responsiveLayout="scroll">
          <slot name="header">
            <div class="table-header">
              Templates
              <Button icon="pi pi-refresh" />
            </div>
          </slot>
          <Column field="id" header="id"></Column>

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
                      <Button onClick={() => onClickUpdateContent(slotProps.data.id)}>
                        Update
                      </Button>,
                      <Button onClick={() => onClickCancelContent(slotProps.data.id)}>
                        Cancel
                      </Button>,
                    ],
                  }}
                </Column>,
              ]}
        </DataTable>
      </div>
    );
  },
});
