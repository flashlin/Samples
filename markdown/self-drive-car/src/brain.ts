import { RadarLineLength } from './gameUtils';
import * as tf from '@tensorflow/tfjs';

class NormalizationLayer extends tf.layers.Layer {
    call(input: tf.Tensor) {
        return tf.div(input, tf.scalar(RadarLineLength));
    }
}

export class Brain {
    model = tf.sequential();
    constructor() {
        const model = this.model;
        // 輸入層，將輸入值正規化到 0~1
        model.add(tf.layers.dense({ inputShape: [5], units: 5, activation: 'relu' }));
        model.add(new NormalizationLayer());
        model.add(tf.layers.dense({ units: 6, activation: 'sigmoid' }));
        model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
        model.compile({ optimizer: 'sgd', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    }

    rewardFunction(state: number[], action: number) {
       // 自定義獎勵邏輯，例如：
       let reward = 0;
       // 你的代碼在這裡...
       return reward;
    }

    async control(radarDistances: number[]) {
        const model = this.model;
        const state = tf.tensor(radarDistances);
        let actionProbabilities = model.predict(state) as tf.Tensor;
        let action = tf.multinomial(actionProbabilities.flatten(), 1).arraySync()[0];

        let array = await action.dataSync();
        let reward = this.rewardFunction(state, action);
        let target = tf.oneHot([action], 4).mul(tf.scalar(reward));
        model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
        await model.fit(state, target, {epochs: 1});
    }
}
