import * as tf from '@tensorflow/tfjs-node';
import { convertId9ToNumbers, generateRandomID, generateTrainData } from './generator';

function normalizeInputs(inputs: number[][]) {
    const inputData = tf.tensor2d(inputs);

    const embedding = tf.layers.embedding({
        inputDim: 10,
        outputDim: 1,
    });

    // Apply the embedding layer
    const embedded = embedding.apply(inputData) as tf.Tensor<tf.Rank>;

    const flattened = embedded.reshape([embedded.shape[0], -1]);

    // Normalize the embeddings to 0-1 range
    // const normalized = tf.div(
    //     flattened,
    //     tf.max(flattened)
    // );

    const min = tf.min(flattened);
    const max = tf.max(flattened);
    const normalized = tf.div(
        tf.sub(flattened, min),
        tf.sub(max, min)
    );

    return normalized;
}

class LogisticRegression {
    private model: tf.Sequential;

    constructor() {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: 10, inputShape: [10], activation: 'softmax' }));
    }

    async train(inputs: number[][], labels: number[], epochs: number = 100) {
        // Convert labels to one-hot encoding, since it's a multi-class classification problem
        const oneHotLabels = labels.map(label => {
            const oneHot = Array(10).fill(0);
            oneHot[label] = 1;
            return oneHot;
        });

        const inputsData = normalizeInputs(inputs);
        const inputTensor = inputsData; //tf.tensor2d(inputs);
        const labelTensor = tf.tensor2d(oneHotLabels);

        // Compile the model
        this.model.compile({
            optimizer: tf.train.adam(),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        // Train the model
        await this.model.fit(inputTensor, labelTensor, { epochs });
    }

    predict(inputs: number[][]): number[] {
        const inputTensor = tf.tensor2d(inputs);
        const predictions = this.model.predict(inputTensor) as tf.Tensor;
        return Array.from(predictions.argMax(-1).dataSync());
    }
}

async function main() {
    // Example data
    //const inputs = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9, 1, 0]];
    //const labels = [1, 2]; // Sample output labels

    const { xTrain, yTrain } = generateTrainData(100000);
    const inputs = xTrain;
    const labels = yTrain;

    // Create and train model
    const logisticRegression = new LogisticRegression();
    await logisticRegression.train(inputs, labels, 200);

    // Make predictions
    //const predictions = logisticRegression.predict(inputs);
    //console.log('Predictions:', predictions);

    for (let i = 0; i < 10; i++) {
        const id = generateRandomID();
        const inputs = convertId9ToNumbers(id);
        const predictions = logisticRegression.predict([inputs]);
        console.log(`ID: ${id} -> ${predictions}`);
    }
}

main();