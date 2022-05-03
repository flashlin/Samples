import { defineComponent, Fragment, reactive } from "vue";
import DataTable from "primevue/datatable";
import Column from "primevue/column";
import Button from "primevue/button";
import InputText from "primevue/inputtext";
import "./main.scss";

export default defineComponent({
  setup(props, { expose }) {
    let state = reactive({
      result: "",
      columns: [
        { field: "code", header: "Code" },
        { field: "name", header: "Name" },
        { field: "category", header: "Category" },
        { field: "quantity", header: "Quantity" },
      ],
      products: [
        {
          code: "PRD001",
          name: "Product 1",
          category: "Category 1",
          quantity: 10,
        },
        {
          code: "PRD002",
          name: "Product 2",
          category: "Category 2",
          quantity: 20,
        },
        {
          code: "PRD003",
          name: "Product 3",
          category: "Category 3",
          quantity: 30,
        },
      ],
    });

    const clickFetchSend = () => {
      state.products.push({
        code: "PRD004",
        name: "Product 4",
        category: "Category 4",
        quantity: 40,
      });

      // let userData = {
      //   id: 123,
      //   name: "flash",
      // };
      // fetch("http://example.com/movies.json", {
      //   method: "POST",
      //   headers: {
      //     "Accept": "application/json",
      //     "Content-Type": "application/json",
      //     //"Authorization": `Bearer ${token}`,
      //   },
      //   body: JSON.stringify(userData),
      // }).then(function (response) {
      //   console.log("fetch resp", response);
      //   state.result = response as any;
      // });
    };

    return () => (
      <div>
        Hello
        <Button
          class="p-button-raised p-button-rounded"
          onClick={clickFetchSend}
        >
          Fetch POST
        </Button>
        result = '{state.result}'
        <div class="p-fluid">
          <div class="p-field">
            <label>First name</label>
            <DataTable value={state.products} responsiveLayout="scroll">
              {state.columns.map((col) => (
                <Column field={col.field} header={col.header} key={col.field} />
              ))}
            </DataTable>
          </div>
          <div class="p-field">
            <label for="lastname1">Last name</label>
            <InputText id="lastname1" type="text" />
          </div>
        </div>
      </div>
    );
  },
});
