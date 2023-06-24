import * as tf from '@tensorflow/tfjs-node';
import { convertId9ToNumbers, generateRandomID, generateTrainData } from './generator';

class MLPClassifier {
    private model: tf.Sequential;

    constructor(inputShape: number, units: number[] = [10]) {
        this.model = tf.sequential();
        for (let i = 0; i < units.length; i++) {
            this.model.add(tf.layers.dense({
                units: units[i],
                activation: i === units.length - 1 ? 'softmax' : 'relu',
                inputShape: i === 0 ? [inputShape] : undefined
            }));
        }
        this.model.compile({
            optimizer: 'adam',
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy']
        });
    }

    async train(X: tf.Tensor, y: tf.Tensor, epochs: number = 100) {
        await this.model.fit(X, y, {
            epochs,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss = ${logs!.loss}`);
                }
            }
        });
    }

    predict(X: tf.Tensor): tf.Tensor {
        return this.model.predict(X) as tf.Tensor;
    }
}

(async () => {
    const { xTrain, yTrain } = generateTrainData(100000);
    const X = tf.tensor2d(xTrain);
    const y = tf.tensor1d(yTrain, 'float32');

    const classifier = new MLPClassifier(10, [10, 20, 3]);
    await classifier.train(X, y, 1000);

    // Predict new data
    for (let i = 0; i < 10; i++) {
        const id = generateRandomID();
        const inputs = convertId9ToNumbers(id);
        const X_new = tf.tensor2d([inputs]);
        const predictions = classifier.predict(X_new);
        //predictions.print();
        const indices = predictions.argMax(1);
        //indices.print();
        console.log(`ID: ${id} -> ${indices.dataSync()[0]}`);
    }
})();
