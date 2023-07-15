<script setup lang="ts">
import { ref, onMounted } from 'vue';
import Chart from 'chart.js/auto';
import { NeuralNetwork } from 'brain.js';

const refCanvas = ref<HTMLCanvasElement>();
const data: { x: number, y: number }[] = [];

function linspace(start: number, end: number, num: number) {
  const step = (end - start) / (num - 1);
  return Array.from({length: num}, (_, i) => start + (i * step));
}

let ys = linspace(0, 4*Math.PI, 500); // y range from 0 to 4π
// for (let y of ys) {
//   let x = Math.sin(y);
//   data.push({x, y});
// }

let xs = Array.from({ length: 100}, (_, i) => i);
console.log(xs)
for(let x of xs){
  let y = Math.asin(x);
  data.push({x, y});
}


const trainData = ys.map(x => ({ input: [x], output: [Math.sin(x)] }));
const net = new NeuralNetwork({
  learningRate: 0.1
});
net.train(trainData);
for (let y of ys) {
  let x = net.run([y]) as unknown as number;
  data.push({x, y});
}


onMounted(() => {
  const canvasElement = refCanvas.value!;
  const ctx = canvasElement.getContext('2d')!;

  new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [{
            label: '座標點',
            data: data,
            backgroundColor: 'rgba(0, 0, 255, 1)' // 設定點的顏色
        }]
    },
    options: {
        scales: {
            x: {
                type: 'linear',
                position: 'bottom'
            },
            y: {
                type: 'linear',
                position: 'left'
            }
        }
    }
  });
});

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
