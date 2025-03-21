import { defineComponent, Fragment, onMounted, reactive } from "vue";
import DataTable from "primevue/datatable";
import Column from "primevue/column";
import Button from "primevue/button";
import InputText from "primevue/inputtext";
import "./main.scss";

import { StockApi } from "@/models/stockApi";

export default defineComponent({
  setup(props, { expose }) {
    let state = reactive({
      result: "",
      columns: [
        { field: "id", header: "id" },
        { field: "tranTime", header: "tranTime" },
        { field: "tranType", header: "tranType" },
        { field: "stockId", header: "stockId" },
        { field: "stockName", header: "stockName" },
        { field: "stockPrice", header: "stockPrice" },
        { field: "numberOfShare", header: "numberOfShare" },
        { field: "handlingFee", header: "handlingFee" },
        { field: "balance", header: "balance" },
      ],
      products: [],
    });

    const clickFetchSend = async () => {
      var api = new StockApi();
      var resp = await api.getTransListAsync();
      state.products = resp;
    };

    onMounted(async () => {
      var api = new StockApi();
      var resp = await api.getTransListAsync();
      state.products = resp;
    });

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
