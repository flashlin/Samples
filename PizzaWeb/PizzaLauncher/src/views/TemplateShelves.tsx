import { defineComponent, onMounted, reactive, ref } from "vue";
import { BannerApi, IBannerTemplateEntity } from "@/models/Api";
import Editor, { IEditorProxy } from "@/components/Editor";
import PreviewFrame from "@/components/PreviewFrame";

export default defineComponent({
  props: {},
  setup(props) {
    const state = reactive({
      columns: [
        { field: "id", header: "id" },
        { field: "templateContent", header: "content" },
      ],
      templateList: [] as IBannerTemplateEntity[],
      content: "123",
      previewContent: "abc"
    });

    const editor = ref<IEditorProxy>();

    const onClickSave = () => {
      let newContent = editor.value!.getContent();
      console.log("content", newContent);
    };

    onMounted(async () => {
      let api = new BannerApi();
      state.templateList = await api.getAllTemplatesAsync();
    });

    return () => (
      <div>
        <button onClick={onClickSave}>Update</button>

        <table
          class="table table-striped table-bordered"
          style="width:100%"
        >
          <thead>
            <tr>
              <th>#</th>
              <th>id</th>
              <th>content</th>
              <th>action</th>
            </tr>
          </thead>
          <tbody>
            {state.templateList.map((item) => (
              <tr>
                <td></td>
                <td>{item.id}</td>
                <td>{item.templateContent}</td>
                <td>
                  <button>Update</button>
                </td>
              </tr>
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

        <Editor content={state.content} ref={editor} />
        <PreviewFrame content={state.previewContent} />

      </div>
    );
  },
});
