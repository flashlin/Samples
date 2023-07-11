import { RadarLineCount, RadarLineLength } from './gameUtils';
import * as tf from '@tensorflow/tfjs';

class NormalizationLayer extends tf.layers.Layer {
    call(input: tf.Tensor) {
        return tf.div(input, tf.scalar(RadarLineLength));
    }
}

export class Brain {
    model = tf.sequential();
    first = true;
    prevState: number[] = [];
    prevAction = 0;

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
        const speedRewardWeight = RadarLineLength / 4;
        const distancePenaltyWeight = 0.3;
        const speed = state[1];
        const distances = state.slice(2, state.length);

        let distanceReward = 0;
        if( distances.every(d => d === 0) )
        {
            distanceReward = RadarLineLength;
        } else {
            const distancePenalty = distances.reduce((a, b) => a + b, 0) / distances.length;
            distanceReward = distancePenaltyWeight * distancePenalty;
        }
        
        const reward = speedRewardWeight * speed + distanceReward;
        console.log(`${reward}`);
        return reward;
    }

    async control(getGameState: () => number[]) {
        const model = this.model;
        
        if( this.first ) {
            const state = getGameState();
            const action = this.predict(state);
            this.prevState = state;
            this.prevAction = action;
            this.first = false;
            return action;
        }

        await this.fit(this.prevState, this.prevAction, getGameState())       
        this.first = true;
        return 5;
    }
    
    predict(state: number[]) {
        const stateTensor = tf.tensor([state]);
        let actionProbabilities = this.model.predict(stateTensor) as tf.Tensor;
        let actions = tf.multinomial(actionProbabilities.flatten(), 1).arraySync();
        let action = actions[0] as number;
        return action;
    }

    async fit(state: number[], action: number, newState: number[]) {
        let reward = this.rewardFunction(newState);
        let target = tf.oneHot([action], 4).mul(tf.scalar(reward));

        const stateTensor = tf.tensor([state]);
        await this.model.fit(stateTensor, target, { epochs: 1 });
        await this.saveModelWeights();
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
