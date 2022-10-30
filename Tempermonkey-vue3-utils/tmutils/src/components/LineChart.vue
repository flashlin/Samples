<script setup lang="ts">
import type { IChartDataRow, IChartDataSet } from '@/models/LineChartModel';
import { Chart, type ChartConfiguration, type ChartData, registerables, LineController, LineElement, PointElement, LinearScale, Title } from 'chart.js'
import { onMounted, reactive, ref, watch } from 'vue';

const props = defineProps<{
  modelValue: IChartDataSet,
}>();

interface LineChartExpose {
  update(chartData: IChartDataSet): void
}
defineExpose<LineChartExpose>({
  update,
})

Chart.register(...registerables);
Chart.register(LineController, LineElement, PointElement, LinearScale,
  Title);

const chartRef = ref<HTMLCanvasElement>();
const colors = ["White", 'Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'];
let chart: Chart;

onMounted(() => {
  const datasets = convertToChartDataset(props.modelValue.data);
  chart = createChart({
    labels: props.modelValue.labels,
    datasets,
  })
});

function convertToChartDataset(value: IChartDataRow[]) {
  return value.map((item) => {
    return {
      label: item.title,
      data: item.data,
      backgroundColor: item.backgroundColor,
    };
  });
}


function createChart(chartData: ChartData) {
  const options: ChartConfiguration = {
    type: 'bar', //'line', 'doughnut',
    data: chartData,
  }
  return new Chart(chartRef.value!, options);
}


function update(value: IChartDataSet) {
  chart.data.labels = value.labels;
  chart.data.datasets = convertToChartDataset(value.data);
  chart!.update();
}

// watch(() => props.modelValue, (first, second) => {
//   console.log(
//     "Watch props.modelValue function called with args:",
//     first,
//     second
//   );
// });
</script>

<template>
  <div>
    <canvas ref="chartRef"></canvas>
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
