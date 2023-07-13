import { RadarLineCount, RadarLineLength } from './gameUtils';
import * as tf from '@tensorflow/tfjs';

class NormalizationLayer extends tf.layers.Layer {
    call(input: tf.Tensor) {
        return tf.div(input, tf.scalar(RadarLineLength));
    }
}

export class Brain {
    model = tf.sequential();

    targetModel = tf.sequential();
    replayMemory: Array<[number[], number, number, number[], boolean]> = [];
    batchSize: number = 10;
    discountFactor: number = 0.9;
    states: number[][] = [];
    rewards: number[] = [];
    training = false;

    first = true;
    prevState: number[] = [];
    prevAction = 0;


    constructor() {
        const model = this.model;
        const inputLength = 3 + RadarLineCount;

        // 輸入層，將輸入值正規化到 0~1
        model.add(tf.layers.dense({ inputShape: [inputLength], units: 32, activation: 'relu' }));
        //model.add(new NormalizationLayer());
        //model.add(tf.layers.dense({ units: 6, activation: 'sigmoid' }));
        //model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
        model.add(tf.layers.dense({ units: 4 }));
        //model.compile({ optimizer: 'sgd', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
        model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

        const targetModel = this.targetModel;
        targetModel.add(tf.layers.dense({ units: 32, inputShape: [inputLength], activation: 'relu' }));
        targetModel.add(tf.layers.dense({ units: 4 }));

        this.loadModelWeights();
    }

    private updateModel(): void {
        const miniBatch = this.sampleMiniBatch();
        const states = miniBatch.map((entry) => entry[0]);
        const actions = miniBatch.map((entry) => entry[1]);
        const rewards = miniBatch.map((entry) => entry[2]);
        const nextStates = miniBatch.map((entry) => entry[3]);
        const dones = miniBatch.map((entry) => entry[4]);

        const qValues = this.model.predict(tf.tensor2d(states)) as tf.Tensor2D;
        const nextQValues = this.targetModel.predict(tf.tensor2d(nextStates)) as tf.Tensor2D;

        const targetQValues = qValues.clone();
        for (let i = 0; i < miniBatch.length; i++) {
            const action = actions[i];
            const reward = rewards[i];
            const done = dones[i];

            let targetQValue: number;
            if (done) {
                targetQValue = reward;
            } else {
                const maxNextQValue = nextQValues.slice([i, 0], [1, nextQValues.shape[1]]).max().dataSync()[0];
                targetQValue = reward + this.discountFactor * maxNextQValue;
            }

            targetQValues.dataSync()[i * targetQValues.shape[1] + action] = targetQValue;
        }

        this.model.fit(tf.tensor2d(states), targetQValues, { epochs: 1, batchSize: this.batchSize });

        qValues.dispose();
        nextQValues.dispose();
        targetQValues.dispose();
    }

    private updateTargetModel(): void {
        this.targetModel.setWeights(this.model.getWeights());
    }

    private sampleMiniBatch(): Array<[number[], number, number, number[], boolean]> {
        const numSamples = Math.min(this.batchSize, this.replayMemory.length);
        const indices = Array.from({ length: numSamples }, (_, i) => i);
        const sampledIndices = indices.sort(() => Math.random() - 0.5).slice(0, numSamples);
        return sampledIndices.map((index) => this.replayMemory[index]);
    }

    rewardFunction(state: number[]) {
        const damaged = state[0];
        if (damaged === 1) {
            return -1000;
        }
        let gpsDistance = state[1];
        const speed = state[2];
        const speedRewardWeight = 10;
        const distances = state.slice(3, state.length);
        const distancesReward = this.calculateDistanceReward(distances);

        if( speed < 0){
            gpsDistance = -gpsDistance;
        }

        let speedReward = speed * speedRewardWeight;
        const reward = speedReward + distancesReward + gpsDistance * 10;
        console.log(`${reward}`);
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
            totalReward += elementReward;
        }

        return totalReward;
    }

    async control(getGameState: () => number[]) {
        const state = getGameState();
        if (!this.first) {
            await this.saveNextStateAsync(state);
        } 
        const action = this.predict(state);
        this.first = false;
        return action;
    }

    predict0(state: number[]) {
        const stateTensor = tf.tensor([state]);
        let actionProbabilities = this.model.predict(stateTensor) as tf.Tensor;
        let actions = tf.multinomial(actionProbabilities.flatten(), 1).arraySync();
        let action = actions[0] as number;
        return action;
    }

    predict(state: number[]): number {
        const input = tf.tensor2d([state]);
        const output = this.model.predict(input) as tf.Tensor2D;
        //const action = output.argMax(1).dataSync()[0];

        let actions = tf.multinomial(output.flatten(), 1).arraySync();
        let action = actions[0] as number;

        input.dispose();
        output.dispose();
        return action;
    }

    async saveNextStateAsync(nextState: number[]) {
        if( this.states.length >= this.batchSize) {
            const states = this.states;
            const rewards = this.rewards;
            for (let i = 0; i < states.length - 1; i++) {
                console.log(`train ${i}`)
                const currentState = states[i];
                const currentReward = rewards[i];
                const nextState = states[i + 1];
                const currentAction = this.predict(currentState);
                await this.trainAsync(currentState, currentAction, currentReward, nextState);
            }
            this.states = [];
            this.rewards = [];
        }
        this.states.push(nextState);
        this.rewards.push(this.rewardFunction(nextState));
    }

    async trainAsync(state: number[], action: number, reward: number, nextState: number[]): Promise<void> {
        const qValues = this.model.predict(tf.tensor2d([state])) as tf.Tensor2D;
        const nextQValues = this.model.predict(tf.tensor2d([nextState])) as tf.Tensor2D;
        const maxNextQValue = nextQValues.max().dataSync()[0];
        const targetQValue = reward + this.discountFactor * maxNextQValue;

        const targetQValues = qValues.dataSync() as any;
        targetQValues[action] = targetQValue;

        const target = tf.tensor2d([targetQValues]);
        await this.model.fit(tf.tensor2d([state]), target, { epochs: 1, batchSize: 1 });
        this.saveModelWeights();

        qValues.dispose();
        nextQValues.dispose();
        target.dispose();
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
        if (weightsJSON == null) {
            return;
        }
        const weights = JSON.parse(weightsJSON);
        this.model.setWeights(weights.map((arr: any) => tf.tensor(arr)));
    }
}
