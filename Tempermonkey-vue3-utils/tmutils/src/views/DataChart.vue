<script setup lang="ts">
import LineChart from "@/components/LineChart.vue";
import type { IChartDataSet } from "@/models/LineChartModel";

const dataset: IChartDataSet = {
  labels: ["2019", "2020", "2021"],
  data: [
    {
      title: "title1",
      data: [1, 2, 3],
      color: "White"
    },
    {
      title: "title2",
      data: [3, 4, 1],
      color: "Red"
    }
  ]
}
</script>

<template>
  <div>
    <LineChart :modelValue="dataset"></LineChart>
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
