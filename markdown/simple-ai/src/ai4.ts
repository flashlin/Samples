import * as tf from '@tensorflow/tfjs';
import { generateRandomID, generateTrainData, convertId9ToNumbers } from './generator';

const state: { weights: number[], biases: number[], lastWeight: number, lastBias: number } = {
    weights: [],
    biases: [],
    lastWeight: Math.random(),
    lastBias: Math.random(),
};

for (let n = 0; n < 10; n++) {
    state.weights.push(Math.random());
    state.biases.push(Math.random());
}


function predictTensorFn() {
    return (...args: tf.Tensor[]) => {
        return tf.tidy(() => {
            const inputsTensor = args[0] as tf.Tensor1D;
            const positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            let weights = tf.variable(tf.tensor1d(state.weights));
            let biases = tf.variable(tf.tensor1d(state.biases));

            const selectedWeights = tf.gather(weights, positions);
            const multiplied = tf.mul(inputsTensor, selectedWeights);

            const selectedBiases = tf.gather(biases, positions);
            const result = tf.add(multiplied, selectedBiases);

            const sum = tf.sum(result);

            let lastWeight = tf.scalar(state.lastWeight);
            let lastBais = tf.scalar(state.lastBias);

            const lastSum = tf.add(tf.mul(sum, lastWeight), lastBais);
            return lastSum;
        });
    };
}




class NeuronNetwork {
    weights: number[] = [];
    biases: number[] = [];
    lastWeight: number = Math.random();
    lastBais: number = Math.random();
    fn = predictTensorFn();

    constructor(count: number) {
        for (let n = 0; n < count; n++) {
            this.weights.push(Math.random());
            this.biases.push(Math.random());
        }
    }

    async predict(inputs: number[], target: number) {
        //console.log(`${inputs} ${inputs.length}`)
        const result = tf.tidy(() => {
            const inputsTensor = tf.tensor1d(inputs);
            const targetTensor = tf.scalar(target);
            const lastSum = this.fn(inputsTensor);
            //-----
            const loss = tf.losses.meanSquaredError(targetTensor, lastSum);
            console.log(`loss = ${loss}`);

            const gradients = tf.grads(this.fn);
            let weights = tf.variable(tf.tensor1d(state.weights));
            let biases = tf.variable(tf.tensor1d(state.biases));
            const [dLossdWeights, dLossdBiases] = gradients([weights, biases]);

            // 使用梯度下降法更新權重和偏差
            const learningRate = 0.1;
            weights.assign(weights.sub(dLossdWeights.mul(learningRate)));
            biases.assign(biases.sub(dLossdBiases.mul(learningRate)));

            state.weights = Array.from(weights.dataSync());
            state.biases = Array.from(biases.dataSync());

            let lastWeight = tf.scalar(state.lastWeight);
            let lastBias = tf.scalar(state.lastBias);
            const [dLossdWeight, dLossdBias] = gradients([lastWeight, lastBias]);
            lastWeight = lastWeight.sub(dLossdWeight.mul(learningRate));
            lastBias = lastBias.sub(dLossdBias.mul(learningRate));
            state.lastWeight = lastWeight.arraySync();
            state.lastBias = lastBias.arraySync();

            return lastSum;
        });
        // 將結果轉換為數字
        const output = await result.data();
        result.dispose();
        return output[0];
    }


    backgrient(result: number, target: number) {
        tf.tidy(() => {
            const resultTensor = tf.scalar(result);
            const targetTensor = tf.scalar(target);
            const loss = tf.losses.meanSquaredError(targetTensor, resultTensor);
            console.log(`loss = ${loss}`);
            const gradients = tf.grads((x: tf.Tensor) => loss.mul(x));

            let weights = tf.variable(tf.tensor1d(this.weights));
            let biases = tf.variable(tf.tensor1d(this.biases));
            const [dLossdWeights, dLossdBiases] = gradients([weights, biases]);

            console.log(`update`)
            // 使用梯度下降法更新權重和偏差
            const learningRate = 0.1;
            weights.assign(weights.sub(dLossdWeights.mul(learningRate)));
            biases.assign(biases.sub(dLossdBiases.mul(learningRate)));

            console.log(`update 2`)
            this.weights = Array.from(weights.dataSync());
            this.biases = Array.from(biases.dataSync());

            let lastWeight = tf.scalar(this.lastWeight);
            let lastBais = tf.scalar(this.lastBais);
            const [dLossdWeight, dLossdBias] = gradients([lastWeight, lastBais]);
            lastWeight = lastWeight.sub(dLossdWeight.mul(learningRate));
            lastBais = lastBais.sub(dLossdBias.mul(learningRate));
        });
    }
}


const { xTrain, yTrain } = generateTrainData(10000);
let model = new NeuronNetwork(10);

async function train() {
    for (let i = 0; i < 10000; i++) {
        await model.predict(xTrain[i], yTrain[i]);
        //model.backgrient(result, yTrain[i]);
    }
}

async function test() {
    console.log('===== test =====');
    for (let i = 0; i < 10; i++) {
        const id = generateRandomID();
        const inputs = convertId9ToNumbers(id);
        const result = await model.predict(inputs, 0);
        console.log(`${id} ${result}`);
    }
}

train();
//test();
