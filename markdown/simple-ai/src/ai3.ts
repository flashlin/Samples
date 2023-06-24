import * as tf from '@tensorflow/tfjs-node';
import { generateRandomID, generateTrainData, getInputs } from './generator';
import { convertIdToNumbers } from './ai';
import path from 'path';
import fs from 'fs';

class LSTMModel {
  private model: tf.Sequential;

  constructor() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 10, inputShape: [10], }));
    //this.model.add(tf.layers.dense({ units: 10, }));
    this.model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  }

  compile() {
    this.model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: 'adam',
      metrics: ['accuracy'],
    });
  }

  // 預測輸入的數字
  predict(inputs: number[][]) {
    const inputTensor = tf.tensor2d(inputs);
    const outputTensor = this.model.predict(inputTensor) as tf.Tensor;
    const predictions = outputTensor.arraySync() as number[][];
    const results: number[] = [];
    for (const prediction of predictions) {
      const predictedIndex = prediction.indexOf(Math.max(...prediction));
      results.push(predictedIndex);
    }
    return results;
  }


  fit(inputs: number[][], targets: number[]) {
    const numClasses = 10;
    const targetTensor = tf.oneHot(tf.tensor1d(targets, 'int32'), numClasses);
    const reshapedTarget = tf.reshape(targetTensor, [targets.length, numClasses]);
    const inputTensor = tf.tensor2d(inputs);
    return this.model.fit(inputTensor, reshapedTarget, { epochs: 500, batchSize: 1 });
  }

  saveModel(path: string) {
    return this.model.save(path);
  }

  loadModel(path: string) {
    return tf.loadLayersModel(path).then(model => {
      this.model = model as tf.Sequential;
    });
  }
}

// 使用範例
async function main() {
  const model = new LSTMModel();

  const modelFile = 'file://./models/model.json';
  if (fs.existsSync(modelFile)) {
    console.log('load model');
    await model.loadModel(modelFile);
  }
  model.compile();

  const { xTrain, yTrain } = generateTrainData(10000);
  await model.fit(xTrain, yTrain);
  const modelPath = 'file://./models'
  await model.saveModel(modelPath);

  for (let i = 0; i < 10; i++) {
    const id = generateRandomID();
    const inputs = getInputs(id);
    const result = model.predict([inputs]);
    console.log(`${id} ${result}`);
  }
}

main().catch(console.error);

// for(let i=0; i<100; i++) {
//   const id = generateRandomID();
//   console.log(id);
// }