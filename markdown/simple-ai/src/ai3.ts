import * as tf from '@tensorflow/tfjs-node';
import { generateRandomID, generateTrainData, convertId9ToNumbers } from './generator';
import { convertIdToNumbers } from './ai';
import path from 'path';
import fs from 'fs';

class CustomModel {
  private model: tf.Sequential;

  constructor() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 11, inputShape: [10] }));
    this.model.add(tf.layers.dense({ units: 3, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  }

  compile() {
    this.model.compile({
      loss: 'sparseCategoricalCrossentropy',
      optimizer: 'adam',
      metrics: ['accuracy'],
    });
  }

  async forward(inputs: number[][]): Promise<number[]> {
    const inputTensor = tf.tensor2d(inputs);
    const predictions = this.model.predict(inputTensor) as tf.Tensor;
    const argMax = await predictions.argMax(-1).data();
    return Array.from(argMax);
  }

  async train(inputs: number[][], targets: number[], epochs: number) {
    const invalidIndex = inputs.findIndex(subArray => subArray.length !== 10);
    //if (inputs.some(subArray => subArray.length !== 10)) {
    if( invalidIndex != -1) {
      throw new Error(`輸入數據的形狀不正確, inputs[${invalidIndex}] 應該應該是長度為 10 的子數組. [${inputs[invalidIndex]}]`);
    }
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 10]);
    const targetTensor = tf.tensor1d(targets, 'float32');
    await this.model.fit(inputTensor, targetTensor, {
      epochs: epochs,
      batchSize: 32,
    });
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
  const model = new CustomModel();

  const modelFile = 'file://./models/model.json';
  if (fs.existsSync(modelFile)) {
    console.log('load model');
    await model.loadModel(modelFile);
  }
  model.compile();

  const { xTrain, yTrain } = generateTrainData(10000);
  await model.train(xTrain, yTrain, 500);
  const modelPath = 'file://./models'
  await model.saveModel(modelPath);

  for (let i = 0; i < 10; i++) {
    const id = generateRandomID();
    const inputs = convertId9ToNumbers(id);
    const result = await model.forward([inputs]);
    console.log(`${id} ${result}`);
  }
}

main();
