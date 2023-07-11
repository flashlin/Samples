import { RadarLineCount, RadarLineLength } from './gameUtils';
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
        model.add(tf.layers.dense({ inputShape: [2 + RadarLineCount], units: 5, activation: 'relu' }));
        //model.add(new NormalizationLayer());
        model.add(tf.layers.dense({ units: 6, activation: 'sigmoid' }));
        model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
        //model.compile({ optimizer: 'sgd', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
        model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        this.loadModelWeights();
    }

    rewardFunction(state: number[]) {
        const damaged = state[0];
        if( damaged === 1 ) {
            return 0;
        }
        const speedRewardWeight = 1;
        const distancePenaltyWeight = 1.2;
        const speed = state[1];
        const distances = state.slice(2, state.length);
        
        const distancePenalty = distances.reduce((a, b) => a + b, 0) / distances.length;
        const reward = speedRewardWeight * speed - distancePenaltyWeight * distancePenalty;
        return reward;
    }

    async control(getGameState: () => number[]) {
        const model = this.model;
        const state = tf.tensor(getGameState());
        let actionProbabilities = model.predict(state) as tf.Tensor;
        let actions = tf.multinomial(actionProbabilities.flatten(), 1).arraySync();
        let action = actions[0] as number;

        const newState = getGameState();
        let reward = this.rewardFunction(newState);
        let target = tf.oneHot([action], 4).mul(tf.scalar(reward));
        await model.fit(state, target, { epochs: 1 });
        await this.saveModelWeights();

        return action;
    }

    async saveModelWeights() {
        const model = this.model;
        const weights = model.getWeights().map(tensor => tensor.arraySync());
        localStorage.setItem('my-model-weights', JSON.stringify(weights));
    }

    async loadModelWeights() {
        const weightsJSON = localStorage.getItem('my-model-weights');
        if( weightsJSON == null ) {
            return;
        }
        const weights = JSON.parse(weightsJSON);
        this.model.setWeights(weights.map((arr: any) => tf.tensor(arr)));
    }
}
