import * as tf from '@tensorflow/tfjs';
import { convertIdToNumbers } from './ai';
import { generateRandomID } from './generator';

class NeuralNetwork {
  private model: tf.Sequential;

  constructor() {
    this.model = this.buildModel();
  }

  private buildModel() {
    let model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, inputShape: [9], activation: 'relu', }));
    //model.add(tf.layers.dense({ units: 5, activation: 'relu', }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: 'sgd',
      metrics: ['accuracy']
    });
    return model;
  }

  async predict(input: number[]): Promise<number> {
    const xs = tf.tensor2d([input]);
    const output = await this.model.predict(xs) as tf.Tensor<tf.Rank.R2>;
    const probabilities = Array.from(output.dataSync());
    //console.log(probabilities)
    const maxProbability = Math.max(...probabilities);
    const maxProbabilityIndex = probabilities.indexOf(maxProbability);
    return maxProbabilityIndex;
  }

  async train(input: number[], label: number[]): Promise<number> {
    const xs = tf.tensor2d([input]);
    const ys = tf.tensor2d([label]);
    const history = await this.model.fit(xs, ys, { epochs: 1 });
    xs.dispose();
    ys.dispose();

    const lossValue = history.history.loss[0] as number;
    return lossValue;
  }

  async train2(input: number[][], label: number[][]): Promise<number> {
    const xs = tf.tensor2d(input);
    const ys = tf.tensor2d(label);
    const loss = tf.losses.softmaxCrossEntropy;
    this.model.compile({ optimizer: 'adam', loss: loss });

    const history = await this.model.fit(xs, ys, { 
      epochs: 500, batchSize:32, verbose: 1,
    });
    xs.dispose();
    ys.dispose();
    const len = history.history.loss.length;
    const lossValue = history.history.loss[len-1] as number;
    return lossValue;
  }

  public saveModel(path: string) {
     this.model.save(`file://${path}`);
  }
}

// 建立神經網路實例
const neuralNetwork = new NeuralNetwork();

// // 向前傳播
// const input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
// const probabilities = neuralNetwork.forwardPropagation(input);
// console.log('向前傳播結果:', probabilities);

// // 向後傳播
// const target = [0, 0, 0, 0, 1, 0, 0, 0, 0];
// const loss = neuralNetwork.backPropagation(input, target);
// console.log('向後傳播損失:', loss);


async function predict(id: string) {
    const idStr = id.substring(0, 9);
    const inputs = convertIdToNumbers(idStr);
    const probabilities = await neuralNetwork.predict(inputs);
    console.log(`${id} checkCode=${probabilities}`);
}


function generateItem() {
    const id = generateRandomID();
    const idStr = id.substring(0, 9);
    const inputs = convertIdToNumbers(idStr);
    const target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const checkCode = id.substring(9).charCodeAt(0) - '0'.charCodeAt(0);
    target[checkCode] = 1;
    return { inputs, target };
}


function generateData(batch: number) {
  const inputsData = [];
  const targetData = [];
  for(let i=0; i<batch; i++) {
    const item = generateItem();
    inputsData.push(item.inputs);
    targetData.push(item.target);
  }
  return { inputsData, targetData };
}

async function train2() {
  let data = generateData(100);
  const history = await neuralNetwork.train2(data.inputsData, data.targetData);
  console.log(history);
}

async function train() {
  for(let epoch=0; epoch<1000; epoch++) {
    const id = generateRandomID();
    const idStr = id.substring(0, 9);
    const inputs = convertIdToNumbers(idStr);
    const target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const checkCode = id.substring(9).charCodeAt(0) - '0'.charCodeAt(0);
    target[checkCode] = 1;
    const history = await neuralNetwork.train(inputs, target);
    console.log(history)
  }
}

async function main() {
  await train2();
  for(let i=0; i<10; i ++) {
    const id = generateRandomID();
    await predict(id);
  }
}
main();