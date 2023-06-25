import * as tf from '@tensorflow/tfjs-node';
import { convertId9ToNumbers, generateRandomID } from './generator';

function normalize1d(inputs: number[]) {
    const inputsTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const inputsMax = inputsTensor.max();
    //const inputsMax = tf.fill([inputs.length, 1], 9);
    const inputsMin = inputsTensor.min();
    //const inputsMin = tf.fill([inputs.length, 1], 0);
    const normalized = inputsTensor.sub(inputsMin).div(inputsMax.sub(inputsMin));
    return normalized;
}

function normalize2d(inputs: number[][]) {
    //const seqLen = inputs[0].length;
    const inputsTensor = tf.tensor2d(inputs);
    const inputsMax = tf.max(inputsTensor);
    //const inputsMax = tf.fill([inputs.length, seqLen], 27);
    const inputsMin = tf.min(inputsTensor);
    //const inputsMin = tf.fill([inputs.length, seqLen], 0);
    const normalized = inputsTensor.sub(inputsMin).div(inputsMax.sub(inputsMin));
    return normalized;
}


class CustomModel {
    private model: tf.Sequential;

    constructor() {
        let model = this.model = tf.sequential();
        model.add(tf.layers.dense({ units: 9, inputShape: [9], }));
        //model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
        model.add(tf.layers.dense({ units: 1, }));
    }

    async train(inputs: number[][], labels: number[], epochs: number = 100) {
        const inputTensor = this.toInputsTensor(inputs);
        const labelTensor = this.toLabelsTensor(labels);

        this.model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse'],
        });

        await this.model.fit(inputTensor, labelTensor, { epochs });
    }

    predict(inputs: number[][]): number[] {
        const inputTensor = this.toInputsTensor(inputs);
        const predictions = this.model.predict(inputTensor) as tf.Tensor<tf.Rank>;
        return Array.from(predictions.argMax(0).dataSync());
        //return Array.from(predictions.dataSync());
    }

    toInputsTensor(inputs: number[][]) {
        const inputTensor = tf.tensor(inputs);
        return inputTensor;
    }

    toLabelsTensor(labels: number[]) {
        const labelTensor = normalize1d(labels);
        return labelTensor;
    }
}

function idToNumbers(id: string) {
    const inputs9 = convertId9ToNumbers(id);
    const inputs: number[] = [];
    let first = 0;
    for (let n = 0; n < inputs9.length; n++) {
        if (n == 0) {
            first = inputs9[0] * 10;
            continue;
        }
        if (n == 1) {
            inputs.push(inputs9[n] + first);
            continue;
        }
        inputs.push(inputs9[n]);
    }
    const code = id.substring(9).charCodeAt(0) - '0'.charCodeAt(0);
    inputs.push(code);
    return inputs;
}

function generateItem() {
    const id = generateRandomID();
    const inputs9 = convertId9ToNumbers(id);
    const inputs: number[] = [];
    let first = 0;
    for (let n = 0; n < inputs9.length; n++) {
        if (n == 0) {
            first = inputs9[0] * 10;
            continue;
        }
        if (n == 1) {
            inputs.push(inputs9[n] + first);
            continue;
        }
        inputs.push(inputs9[n]);
    }
    const target = id.substring(9).charCodeAt(0) - '0'.charCodeAt(0);

    if (inputs.length != 9) {
        throw new Error(`${id} inputs=${inputs} t=${target}`);
    }

    return { inputs, target };
}

export function generateTrainData(dataSize: number) {
    let xTrain = [];
    let yTrain = [];
    for (let n = 0; n < dataSize; n++) {
        const item = generateItem();
        xTrain.push(item.inputs);
        yTrain.push(item.target);
    }
    return { xTrain, yTrain };
}

async function main() {
    const model = new CustomModel();

    console.log(`start training`);
    const items = generateTrainData(10000);
    await model.train(items.xTrain, items.yTrain, 500);

    //const predictions = await model.predict(items.xTrain);
    //console.log('pred=', predictions);

    for (let i = 0; i < 10; i++) {
        const id = generateRandomID();
        const inputs = idToNumbers(id).slice(0, 9);
        const predictions = await model.predict([inputs]);
        console.log(`ID: ${id} -> ${predictions}`);
    }
}

main();