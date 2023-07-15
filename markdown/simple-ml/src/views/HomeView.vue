<script setup lang="ts">
import { ref, onMounted } from 'vue';
import Chart from 'chart.js/auto';

const refCanvas = ref<HTMLCanvasElement>();

const xValues: number[] = [];
const yValues: number[] = [];

// 計算方程式的值並填充到陣列中
for (let x = -10; x <= 10; x++) {
  xValues.push(x);
  yValues.push(x + 3);
}

onMounted(() => {
  const canvasElement = refCanvas.value!;
  const ctx = canvasElement.getContext('2d')!;

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: xValues,
      datasets: [{
        label: 'y = x + 3',
        data: yValues,
        borderColor: 'blue',
        backgroundColor: 'transparent',
      }],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: 'x',
          },
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'y',
          },
        },
      },
    },
  });
});

// 使用 Chart.js 繪製圖表


</script>

<template>
  <div class="container">
    <canvas ref="refCanvas" />
  </div>
</template>

<style scoped>
.container {
  position: relative;
  text-align: center;
}

.title {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 8em;
  font-weight: bold;
  background: linear-gradient(to right, blue, purple);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>
