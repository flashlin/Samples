<script setup lang="ts">
import LineChart from "@/components/LineChart.vue";
import type { IChartDataRow, IChartDataSet } from "@/models/LineChartModel";
import CodeEditor from "@/components/CodeEditor.vue";
import { reactive, ref } from "vue";
import type { IDataChartData } from "@/models/DataChartModel";
import Button from 'primevue/button';
import InputText from 'primevue/inputtext';
import Textarea from "primevue/textarea";

const data = reactive<IDataChartData>({
  code: ``,
  csvJson: `[{"date":"2019","price1":"2","price2":"2"},{"date":"2020","price1":"1","price2":"3"},{"date":"2021","price1":"5","price2":"4"}]`,
});

const chartRef = ref<InstanceType<typeof LineChart>>();

const chartDataset: IChartDataSet = reactive({
  labels: ["x1", "x2", "x3"],
  data: [
    {
      title: "title1",
      data: [1, 2, 3],
      backgroundColor: "White"
    },
    {
      title: "title2",
      data: [3, 4, 1],
      backgroundColor: "Red"
    }
  ]
});

function onGenerateChart() {
  let objArr: any[] = JSON.parse(data.csvJson);
  let labels: string[] = [];
  let titles: string[] = [];

  let dict: Record<string, IChartDataRow> = {};

  objArr.forEach((obj, index) => {
    let keys = Object.keys(obj);
    if (index == 0) {
      keys.slice(1).forEach((title) => {
        titles.push(title);
        dict[title] = {
          title: title,
          data: [],
          backgroundColor: "Red",
        };
      });
    }

    labels.push(obj[keys[0]]);
    keys.slice(1).forEach((key, idx) => {
      let chartData = dict[key];
      chartData.data.push(obj[key]);
    });
  });


  let dataset = [];
  for (let key in dict) {
    dataset.push(dict[key]);
  }

  chartDataset.labels = labels;
  chartDataset.data = dataset;
  chartRef.value!.update(chartDataset);
}

</script>

<template>
  <div>
    <Textarea :modelValue="data.csvJson" rows="10" cols="80" />
    <InputText type="text" />
    <Button :onclick="onGenerateChart">Generate</Button>
    <LineChart :modelValue="chartDataset" ref="chartRef"></LineChart>
  </div>
</template>

<style scoped>
h1 {
  font-weight: 500;
  font-size: 2.6rem;
  top: -10px;
}

h3 {
  font-size: 1.2rem;
}

.greetings h1,
.greetings h3 {
  text-align: center;
}

@media (min-width: 1024px) {

  .greetings h1,
  .greetings h3 {
    text-align: left;
  }
}
</style>
