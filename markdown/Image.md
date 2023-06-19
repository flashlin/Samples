

Test

```javascript {cmd="node"}
const date = Date.now()
console.log(date.toString())
```



```javascript {cmd="node"}
class NeuralNetwork {
  constructor() {
    this.weight = Math.random();
    this.bias = Math.random();
  }

  forward(inputs) {
    const neuron = this.sigmoid(inputs[0] * this.weight + this.bias);
    return [neuron];
  }

  backward(inputs, targets, learningRate) {
    const [neuron] = this.forward(inputs);
    // 計算損失
    const loss1 = neuron - targets[0];

    // 更新权重和偏置
    this.weight -= inputs[0] * loss1 * learningRate;
    this.bias -= loss1 * learningRate;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
}

const neuralNetwork = new NeuralNetwork();
const inputs = [0.5, 0.8];
const targets = [0.3, 0.6];
const learningRate = 0.1;
const outputs = neuralNetwork.forward(inputs);
console.log("Output from neuron 1:", outputs[0]);
neuralNetwork.backward(inputs, targets, learningRate);
```