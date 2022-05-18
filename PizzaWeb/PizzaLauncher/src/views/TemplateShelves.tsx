import { defineComponent, Fragment, onMounted, reactive, Ref, ref } from "vue";
import { BannerApi, IBannerTemplateEntity } from "@/models/Api";
import PreviewFrame from "@/components/PreviewFrame";
import Editor, { IEditorProxy } from "@/components/Editor";

export default defineComponent({
  props: {},
  setup(props) {
    const state = reactive({
      columns: [
        { field: "id", header: "id" },
        { field: "templateContent", header: "content" },
      ],
      templateList: [] as IBannerTemplateEntity[],
      templateShortContent: [] as string[],
      currentEditId: "",
      previewContent: "abc",
    });

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

    const onClickEdit = (id: string) => {
      let item = state.templateList.find((x) => x.id == id)!;
      let idx = state.templateList.indexOf(item);

      let c = editorRefs.value[idx].getContent();
      console.log("c", c);
      //state.itemRefs[idx].value.getContent();
      //state.previewContent = item.templateContent;
      //state.editContent = item.templateContent;
      state.currentEditId = id;
    };

    const onClickSave = () => {
      let newContent = editor.value!.getContent();
      console.log("content", newContent);
    };

    onMounted(async () => {
      let api = new BannerApi();
      state.templateList = await api.getAllTemplatesAsync();
      // state.templateList = templateList.map(x => {
      //   let newObj = new BannerTemplateData(x);
      //   newObj.templateShortContent = subContent(x.templateContent);
      //   return newObj;
      // });
      for(var item of state.templateList) {
        state.templateShortContent.push(subContent(item.templateContent));
      }
    });

    return () => (
      <div>
        <button onClick={onClickSave}>Update</button>
        <i class="bi bi-clipboard"></i>

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
                  <td>{subContent(item.templateContent)}</td>
                  <td>
                    <button onClick={() => onClickEdit(item.id)}>Edit</button>
                    <button>Update</button>
                  </td>
                </tr>
                <tr v-show={state.currentEditId == item.id}>
                  <td colspan='3'>
                    <Editor content={item.templateContent} ref={setItemRef} />
                  </td>
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
