import * as tf from '@tensorflow/tfjs-node';

class MLPClassifier {
    private model: tf.Sequential;

    constructor(inputShape: number, units: number[] = [10], optimizer: string = 'adam', loss: string = 'sparseCategoricalCrossentropy') {
        this.model = tf.sequential();
        for (let i = 0; i < units.length; i++) {
            this.model.add(tf.layers.dense({
                units: units[i],
                activation: i === units.length - 1 ? 'softmax' : 'relu',
                inputShape: i === 0 ? [inputShape] : undefined
            }));
        }
        this.model.compile({ optimizer, loss, metrics: ['accuracy'] });
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
    // Example data
    const X = tf.tensor2d([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9, 1, 0],
        // ... more data
    ]);

    const y = tf.tensor1d([1, 2], 'float32'); // labels must be integers for sparseCategoricalCrossentropy

    // Instantiate and train the model
    const classifier = new MLPClassifier(10, [10, 10]);
    await classifier.train(X, y, 50);

    // Predict new data
    const X_new = tf.tensor2d([
        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    ]);
    const predictions = classifier.predict(X_new);
    //predictions.print();
    const indices = predictions.argMax(1);
    indices.print();
})();
