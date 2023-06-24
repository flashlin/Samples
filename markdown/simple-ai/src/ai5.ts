import * as tf from '@tensorflow/tfjs';

class SimpleModel {
    private model: tf.Sequential;

    constructor() {
        // 建立一個簡單的模型
        this.model = tf.sequential();

        // 加入第一層 dense
        this.model.add(tf.layers.dense({units: 16, inputShape: [10], activation: 'relu'}));

        // 加入第二層 dense
        this.model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

        // 編譯模型
        this.model.compile({
            optimizer: 'sgd',
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy'],
        });
    }

    // 定義前向傳播
    public async forward(input: number[][]): Promise<number[]> {
        const predictions = this.model.predict(tf.tensor2d(input)) as tf.Tensor;
        const argMax = await predictions.argMax(-1).data();
        return Array.from(argMax);
    }

    // 定義後向傳播與訓練
    public async train(input: number[][], labels: number[], epochs: number): Promise<void> {
        await this.model.fit(tf.tensor2d(input), tf.tensor1d(labels, 'float32'), {
            epochs: epochs
        });
    }

    // 保存模型參數
    public async saveModel(path: string): Promise<void> {
        await this.model.save(path);
    }

    // 載入模型參數
    public async loadModel(path: string): Promise<void> {
        //this.model = await tf.loadLayersModel(path);
    }
}

(async () => {
    // 初始化模型
    const simpleModel = new SimpleModel();

    // 訓練資料
    const input = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9, 1, 0]];
    const labels = [1, 2];

    // 訓練模型
    await simpleModel.train(input, labels, 10);

    // 預測
    const predictions = await simpleModel.forward(input);
    console.log(predictions);

    // 保存模型
    //await simpleModel.saveModel('file://./simple-model');

    // 載入模型
    //await simpleModel.loadModel('file://./simple-model');

    // 再次預測
    const predictionsAfterLoad = await simpleModel.forward(input);
    console.log(predictionsAfterLoad);
})();
