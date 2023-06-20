
```vega-lite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {
    "values": [
      {"x": 1, "y": 1},
      {"x": 2, "y": 2},
      {"x": 3, "y": 3},
      {"x": 5, "y": 5}
    ]
  },
  "layer": [
    {
      "mark": {"type": "point", "filled": true},
      "encoding": {
        "x": {"field": "x", "type": "quantitative"},
        "y": {"field": "y", "type": "quantitative"},
        "color": {"value": "steelblue"}
      }
    },
    {
      "mark": {"type": "line"},
      "encoding": {
        "x": {"field": "x", "type": "quantitative"},
        "y": {"field": "y", "type": "quantitative"}
      }
    }
  ]
}
```



```javascript {cmd="node"}
class Neural {
  constructor() {
    this.weight = Math.random();
    this.bias = Math.random();
  }

  forward(input) {
    const neuron = this.sigmoid(input * this.weight + this.bias);
    return neuron;
  }

  backward(input, output, target, learningRate) {
    const error = target - output;
    this.weight += input * error * learningRate;
    this.bias += error * learningRate;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
}

const neuron = new Neuron();
const inputs  = [1, 2, 3];
const outputs = [];

const output = neuron.forward(input);

// 向後傳播更新權重和偏差
const learningRate = 0.1;
neuron.backward(input, output, target, learningRate);
```

```javascript cmd="node"}
class Neuron {
  constructor() {
    // 初始化權重和偏差
    this.weights = [];
    this.bias = 0;
  }

  forward(input) {
    // 向前傳播
    let sum = 0;
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * this.weights[i];
    }
    sum += this.bias;
    return this.activationFunction(sum);
  }

  backward(input, output, target, learningRate) {
    // 向後傳播
    const error = target - output;
    for (let i = 0; i < input.length; i++) {
      this.weights[i] += input[i] * error * learningRate;
    }
    this.bias += error * learningRate;
  }

  activationFunction(x) {
    // 激活函式（例如 sigmoid 函式）
    return 1 / (1 + Math.exp(-x));
  }
}
```


身分證驗證碼

```javascript {cmd="node"}
function convertIdToNumbers(id) {
  let a = id.substr(0, 1).charCodeAt() - 'A'.charCodeAt() + 10;
  let remainder = id.substr(1);
  let numberText = a + remainder;
  return numberText;
}

function calculateChecksum(id) {
  let weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1];
  let sum = 0;
  let idNumber = convertIdToNumbers(id)

  for (var i = 0; i < 10; i++) {
    num = idNumber.substr(i, 1);
    n = num.charCodeAt() - '0'.charCodeAt();
    sum += n * weights[i];
  }
  let checksum = 10 - sum % 10;
  return checksum;
}

var id = "A123456789";
var checksum = calculateChecksum(id);
console.log(checksum);
```
