import * as tf from '@tensorflow/tfjs-node';
import { convertId9ToNumbers, generateRandomID, generateTrainData } from './generator';

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

        const inputTensor = tf.tensor2d(inputs);
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