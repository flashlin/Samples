export function program(x: number) {
    console.log(x)
    return x;
}


export function convertIdToNumbers(idStr: string) {
    let numbers = [];
    let a = idStr.substring(0, 1).charCodeAt(0) - 'A'.charCodeAt(0) + 10;
    numbers.push(a);

    for (let i = 1; i < idStr.length; i++) {
        let numStr = idStr.substring(i, i + 1);
        let num = numStr.charCodeAt(0) - '0'.charCodeAt(0);
        numbers.push(num);
    }
    return numbers;
}

export class Neuron {
    weight = Math.random();
    bias = Math.random();

    forward(input: number) {
        return input * this.weight + this.bias;
    }

    backward(input: number, output: number, target: number, learningRate: number) {
        const loss = target - output;
        //console.log(`loss=${loss}`)
        this.weight += input * loss * learningRate;
        this.bias += loss * learningRate;
    }
}

function sigmoid(x: number) {
    return 1 / (1 + Math.exp(-x));
}

function round(n: number) {
    return n
    const maxValue = 1000.0e16; 
    const minValue = -100.0e16; 
    n = Math.min(n, maxValue);
    n = Math.max(n, minValue); 
    return n;
}

export class Neuron2 {
    weights: number[];
    biases : number[];

    constructor(n: number) {
        this.weights = [];
        this.biases = [];
        for (let i = 0; i < n; i++) {
            this.weights.push(Math.random());
            this.biases.push(Math.random());
        }
    }

    load(json: string) {
        const obj = JSON.parse(json);
        this.weights = obj.weights;
        this.biases = obj.biases;
    }


    forward(inputs: number[]) {
        let sum = inputs.reduce((sum, input, i) => sum + input * this.weights[i] + this.biases[i], 0);
        return round(sum);
    }

    backward(inputs: number[], output: number, target: number, learningRate: number) {
        const error = round(target - output);
        console.log(`${target} ${output} ${error}`);
        for (let i = 0; i < 9; i++) {
           const input = inputs[i];
           this.weights[i] += learningRate * error * input;
           this.biases[i] += learningRate * error;
           this.weights[i] = round(this.weights[i]);
           this.biases[i] = round(this.biases[i]);
        }
    }

    toJSON() {
        return JSON.stringify({
            'weights': this.weights,
            'biases': this.biases
        });
    }
}

export function train(inputs: number[][], targets: number[], neuron: Neuron2) {
    for (let epoch = 0; epoch < 1000; epoch++) {
        for (let n = 0; n < inputs.length; n++) {
            const idNumbers = inputs[n];
            const target = targets[n];
            const output = neuron.forward(idNumbers);
            const learningRate = 0.001;
            neuron.backward(idNumbers, output, target, learningRate);
        }
    }
    console.log(neuron.toJSON());
}

export function MyId(id: string) {
    const neuron = new Neuron2(9);
    neuron.load(`{"weights":[-0.020684901196313846,0.6565113976980774,0.2597811426120442,-0.24707663342176275,-0.09072944794725564,-0.14906440255583717,-0.036164375777767585,-0.017334529407265882,-0.083732740758375,0.7142098072416383],"biases":[0.28651477721625507,-0.10111395830952205,0.6465506213063712,0.7006215192479774,0.5494372036318627,0.3583326952803101,0.10972676889717001,-0.11921740506205653,0.440472026277425,0.3667029727454545]}`);

    const idStr = id.substring(0, id.length-1);
    const idNumbers = convertIdToNumbers(idStr);
    const checkSum = neuron.forward(idNumbers);
    console.log(`${id} checksum=${checkSum}`);
}
