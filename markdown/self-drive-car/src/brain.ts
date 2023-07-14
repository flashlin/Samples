import { RadarLineCount, RadarLineLength } from './gameUtils';
import * as tf from '@tensorflow/tfjs';

class NormalizationLayer extends tf.layers.Layer {
    call(input: tf.Tensor) {
        return tf.div(input, tf.scalar(RadarLineLength));
    }
}

function toTensor2dFloat32(state: number[]){
    const state16 = new Float32Array(state);
    state = Array.from(state16);
    const input = tf.tensor2d([state]);
    return input;
}

export interface IBrain {
    loadModelWeights(): void;
    predict(state: number[]): number;
    fitAsync(currentState: number[], action: number, nextState: number[], reward: number): Promise<void>;
}

export class TensorflowBrain implements IBrain {
    model = tf.sequential();
    targetModel = tf.sequential();
    replayMemory: Array<[number[], number, number, number[], boolean]> = [];
    batchSize: number = 32;
    discountFactor: number = 0.9;
    states: number[][] = [];
    actions: number[] = [];
    rewards: number[] = [];
    training = false;
    learningRate: number = 0.001;

    first = true;
    prevState: number[] = [];
    prevAction = 0;

    constructor() {
        const model = this.model;
        const inputLength = 2 + RadarLineCount;

        // 輸入層，將輸入值正規化到 0~1
        model.add(tf.layers.dense({ inputShape: [inputLength], units: 32, activation: 'relu', dtype: 'float32' }));
        //model.add(new NormalizationLayer());
        //model.add(tf.layers.dense({ units: 6, activation: 'sigmoid' }));
        //model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
        model.add(tf.layers.dense({ units: 3 , dtype: 'float32' }));
        //model.compile({ optimizer: 'sgd', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
        //model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        model.compile({ loss: 'meanSquaredError', optimizer: tf.train.adam(this.learningRate) });
    

        const targetModel = this.targetModel;
        targetModel.add(tf.layers.dense({ units: 32, inputShape: [inputLength], activation: 'relu', dtype: 'float32' }));
        targetModel.add(tf.layers.dense({ units: 3, dtype: 'float32'}));

        this.loadModelWeights();
    }

    predict(state: number[]): number {
        const input = toTensor2dFloat32(state);
        const output = this.model.predict(input) as tf.Tensor2D;
        const action = output.argMax(1).dataSync()[0];
        input.dispose();
        output.dispose();
        return action;
    }

    async fitAsync(state: number[], action: number, nextState: number[], reward: number): Promise<void> {
        const inputState = toTensor2dFloat32(state);
        const qValues = this.model.predict(inputState) as tf.Tensor2D;

        const nextStateTensor = toTensor2dFloat32(nextState);
        const nextQValues = this.model.predict(nextStateTensor) as tf.Tensor2D;
        const maxNextQValue = nextQValues.max().dataSync()[0];
        const targetQValue = reward + this.discountFactor * maxNextQValue;

        const targetQValues = qValues.dataSync() as any;
        targetQValues[action] = targetQValue;

        const target = tf.tensor2d([targetQValues]);
        await this.model.fit(inputState, target, { epochs: 1, batchSize: 1 });
        this.saveModelWeights();

        inputState.dispose();
        nextStateTensor.dispose();

        qValues.dispose();
        nextQValues.dispose();
        target.dispose();
    }

    saveModelWeights() {
        const model = this.model;
        const weights = model.getWeights().map(tensor => tensor.arraySync());
        localStorage.setItem('my-model-weights', JSON.stringify(weights));
    }

    async loadModelWeights() {
        const weightsJSON = localStorage.getItem('my-model-weights');
        if (weightsJSON == null) {
            return;
        }
        const weights = JSON.parse(weightsJSON);
        this.model.setWeights(weights.map((arr: any) => tf.tensor(arr)));
    }
}

export class Brain {
    first = true;
    model: IBrain;

    prevState: number[] = [];
    prevAction: number = 0;

    constructor() {
        this.model = new TensorflowBrain();
        this.model.loadModelWeights();
    }

    async control(getGameState: () => number[]) {
        const state = getGameState();
        if (!this.first) {
            const nextState = state;
            await this.saveNextStateAsync(this.prevState, this.prevAction, nextState);
        } 
        const action = this.model.predict(state);
        this.prevState = state;
        this.prevAction = action;
        this.first = false;
        return action;
    }

    rewardFunction(state: number[]) {
        const damaged = state[0];
        if (damaged === 1) {
            return -1000;
        }
        let speed = state[1];
        const speedRewardWeight = 10;
        const distances = state.slice(2, state.length);
        const distancesReward = this.calculateDistanceReward(distances);

        if( speed == 0) {
            speed = -100;
        }

        let speedReward = speed * speedRewardWeight;
        const reward = speedReward + distancesReward;
        return reward;
    }

    calculateDistanceReward(distance: number[]) {
        let totalReward = 0;

        for (let i = 1; i < distance.length; i += 2) {
            if (i + 1 < distance.length) {
                const diff = Math.abs(distance[i] - distance[i + 1]);
                const reward = diff > 0 ? -1.1 * diff : RadarLineLength;
                totalReward += reward;
            }
        }

        for (let i = 0; i < distance.length; i++) {
            const elementReward = distance[i] > 0 ? distance[i] : -RadarLineLength;
            if( distance[i] < 50 ) {
                totalReward -= 100;
            }
            totalReward += elementReward;
        }

        return totalReward;
    }

    async saveNextStateAsync(currentState: number[], action: number, nextState: number[]) {
        const reward = this.rewardFunction(nextState);
        await this.model.fitAsync(currentState, action, nextState, reward);
    }
}