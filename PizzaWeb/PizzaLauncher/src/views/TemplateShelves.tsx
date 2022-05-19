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

export default defineComponent({
  props: {},
  template: `
    <h1>Vue 3 TypeScript Template</h1>
    <div class="app">
      <HelloWorld />
    </div>
  `,
  setup(props) {
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

    const editorRefs = ref([] as IEditorProxy[]);
    const editor = ref<IEditorProxy>();

    const setItemRef = (el: any) => {
      editorRefs.value.push(el);
    };

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

    const onClickReload = async () => {
      let api = new BannerApi();
      let resp = await api.getAllTemplatesAsync();
      state.templateList = resp;
    };

    const onClickEdit = (id: string) => {
      //let item = state.templateList.find((x) => x.id == id)!;
      state.currentEditId = id;
    };

    const onClickUpdateContent = async (id: string) => {
      let item = state.templateList.find((x) => x.id == id)!;
      let idx = state.templateList.indexOf(item);
      let newContent = editorRefs.value[idx].getContent();
      item.templateContent = newContent;
      //let resp = await BannerApi.updateBannerTemplate(item);
      state.currentEditId = "";
    };

    const onClickCancelContent = (id: string) => {
      let item = state.templateList.find((x) => x.id == id)!;
      let idx = state.templateList.indexOf(item);
      state.currentEditId = "";
    };

    const onClickSave = () => {
      let newContent = editor.value!.getContent();
      console.log("content", newContent);
    };

    onMounted(async () => {
      let api = new BannerApi();
      let resp = await api.getAllTemplatesAsync();
      state.templateList = resp;
    });

    return () => (
      <div>
        <button onClick={onClickReload}>Reload</button>

        <table class="table table-striped table-bordered" style="width:100%">
          <thead>
            <tr>
              <th>#</th>
              <th>id</th>
              <th>content</th>
              <th>action</th>
            </tr>
          </thead>
          <tbody>
            {state.templateList.map((item, idx) => (
              <Fragment>
                <tr>
                  <td></td>
                  <td>{item.id}</td>
                  <td v-show={!isEditing()}>
                    {subContent(item.templateContent)}
                  </td>
                  <td v-show={!isEditing()}>
                    <button onClick={() => onClickEdit(item.id)}>Edit</button>
                    <button>Update</button>
                  </td>
                  <Fragment>
                    <td v-show={state.currentEditId == item.id}>
                      <Editor content={item.templateContent} ref={setItemRef} />
                    </td>
                    <td v-show={state.currentEditId == item.id}>
                      <button onClick={() => onClickUpdateContent(item.id)}>
                        Update
                      </button>
                      <button onClick={() => onClickCancelContent(item.id)}>
                        Cancel
                      </button>
                    </td>
                  </Fragment>
                </tr>
              </Fragment>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th></th>
              <th>id</th>
              <th>content</th>
              <th>action</th>
            </tr>
          </tfoot>
        </table>

        <PreviewFrame
          content={state.previewContent}
          style={`with:100px; height:100px;`}
        />
      </div>
    );
  },
});
