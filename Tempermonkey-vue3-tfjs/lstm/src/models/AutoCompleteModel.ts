import * as tf from "@tensorflow/tfjs";

export class TextData {
  dataIdentifier_: string;
  textString_: string;
  textLen_: number;
  sampleLen_: number;
  sampleStep_: number;
  charSetSize_ = 0;
  exampleBeginIndices_: any;
  examplePosition_: any;
  indices_: any;
  charSet_: any;

  /**
   * Constructor of TextData.
   * @param {string} dataIdentifier An identifier for this instance of TextData.
   * @param {string} textString The training text data.
   * @param {number} sampleLen Length of each training example, i.e., the input
   *   sequence length expected by the LSTM model.
   * @param {number} sampleStep How many characters to skip when going from one
   *   example of the training data (in `textString`) to the next.
   */
  constructor(
    dataIdentifier: string,
    textString: string,
    sampleLen: number,
    sampleStep: number
  ) {
    tf.util.assert(
      sampleLen > 0,
      () => `Expected sampleLen to be a positive integer, but got ${sampleLen}`
    );
    tf.util.assert(
      sampleStep > 0,
      () =>
        `Expected sampleStep to be a positive integer, but got ${sampleStep}`
    );

    if (!dataIdentifier) {
      throw new Error("Model identifier is not provided.");
    }

    this.dataIdentifier_ = dataIdentifier;

    this.textString_ = textString;
    this.textLen_ = textString.length;
    this.sampleLen_ = sampleLen;
    this.sampleStep_ = sampleStep;

    this.getCharSet_();
    this.convertAllTextToIndices_();
  }

  /**
   * Get data identifier.
   * @returns {string} The data identifier.
   */
  dataIdentifier(): string {
    return this.dataIdentifier_;
  }

  /**
   * Get length of the training text data.
   * @returns {number} Length of training text data.
   */
  textLen() {
    return this.textLen_;
  }

  /**
   * Get the length of each training example.
   */
  sampleLen() {
    return this.sampleLen_;
  }

  /**
   * Get the size of the character set.
   *
   * @returns {number} Size of the character set, i.e., how many unique
   *   characters there are in the training text data.
   */
  charSetSize() {
    return this.charSetSize_;
  }

  /**
   * Generate the next epoch of data for training models.
   *
   * @param {number} numExamples Number examples to generate.
   * @returns {[tf.Tensor, tf.Tensor]} `xs` and `ys` Tensors.
   *   `xs` has the shape of `[numExamples, this.sampleLen, this.charSetSize]`.
   *   `ys` has the shape of `[numExamples, this.charSetSize]`.
   */
  nextDataEpoch(numExamples: number) {
    this.generateExampleBeginIndices_();

    if (numExamples == null) {
      numExamples = this.exampleBeginIndices_.length;
    }

    const xsBuffer = tf.buffer([
      numExamples,
      this.sampleLen_,
      this.charSetSize_,
    ]);
    const ysBuffer = tf.buffer([numExamples, this.charSetSize_]);
    for (let i = 0; i < numExamples; ++i) {
      const beginIndex =
        this.exampleBeginIndices_[
          this.examplePosition_ % this.exampleBeginIndices_.length
        ];
      for (let j = 0; j < this.sampleLen_; ++j) {
        xsBuffer.set(1, i, j, this.indices_[beginIndex + j]);
      }
      ysBuffer.set(1, i, this.indices_[beginIndex + this.sampleLen_]);
      this.examplePosition_++;
    }
    return [xsBuffer.toTensor(), ysBuffer.toTensor()];
  }

  /**
   * Get the unique character at given index from the character set.
   *
   * @param {number} index
   * @returns {string} The unique character at `index` of the character set.
   */
  getFromCharSet(index: number) {
    return this.charSet_[index];
  }

  /**
   * Convert text string to integer indices.
   *
   * @param {string} text Input text.
   * @returns {number[]} Indices of the characters of `text`.
   */
  textToIndices(text: string): number[] {
    const indices = [];
    for (let i = 0; i < text.length; ++i) {
      indices.push(this.charSet_.indexOf(text[i]));
    }
    return indices;
  }

  /**
   * Get a random slice of text data.
   *
   * @returns {[string, number[]} The string and index representation of the
   *   same slice.
   */
  getRandomSlice(): [string, number[]] {
    const startIndex = Math.round(
      Math.random() * (this.textLen_ - this.sampleLen_ - 1)
    );
    const textSlice = this.slice_(startIndex, startIndex + this.sampleLen_);
    return [textSlice, this.textToIndices(textSlice)];
  }

  /**
   * Get a slice of the training text data.
   *
   * @param {number} startIndex
   * @param {number} endIndex
   * @param {bool} useIndices Whether to return the indices instead of string.
   * @returns {string | Uint16Array} The result of the slicing.
   */
  slice_(startIndex: number, endIndex: number) {
    return this.textString_.slice(startIndex, endIndex);
  }

  /**
   * Get the set of unique characters from text.
   */
  getCharSet_() {
    this.charSet_ = [];
    for (let i = 0; i < this.textLen_; ++i) {
      if (this.charSet_.indexOf(this.textString_[i]) === -1) {
        this.charSet_.push(this.textString_[i]);
      }
    }
    this.charSetSize_ = this.charSet_.length;
  }

  /**
   * Convert all training text to integer indices.
   */
  convertAllTextToIndices_() {
    this.indices_ = new Uint16Array(this.textToIndices(this.textString_));
  }

  /**
   * Generate the example-begin indices; shuffle them randomly.
   */
  generateExampleBeginIndices_() {
    // Prepare beginning indices of examples.
    this.exampleBeginIndices_ = [];
    for (
      let i = 0;
      i < this.textLen_ - this.sampleLen_ - 1;
      i += this.sampleStep_
    ) {
      this.exampleBeginIndices_.push(i);
    }

    // Randomly shuffle the beginning indices.
    tf.util.shuffle(this.exampleBeginIndices_);
    this.examplePosition_ = 0;
  }
}

function createDense(alpha_len: number) {
  const layer = tf.sequential();
  layer.add(tf.layers.dense({ units: alpha_len, activation: "softmax" }));
  layer.add(tf.layers.dropout({ rate: 0.2 }));
  return layer;
}

async function create_model1(max_len: number, alpha_len = 26) {
  const model = tf.sequential();
  model.add(
    tf.layers.lstm({
      units: alpha_len * 2,
      inputShape: [max_len, alpha_len],
      dropout: 0.2,
      recurrentDropout: 0.2,
      useBias: true,
      returnSequences: true,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.timeDistributed({
      layer: createDense(alpha_len),
    })
  );
  model.summary();
  return model;
}

export function createModel(
  sampleLen: number,
  charSetSize: number,
  lstmLayerSizes: number[]
) {
  if (!Array.isArray(lstmLayerSizes)) {
    lstmLayerSizes = [lstmLayerSizes];
  }

  const model = tf.sequential();
  for (let i = 0; i < lstmLayerSizes.length; ++i) {
    const lstmLayerSize = lstmLayerSizes[i];
    model.add(
      tf.layers.lstm({
        units: lstmLayerSize,
        returnSequences: i < lstmLayerSizes.length - 1,
        inputShape: i === 0 ? [sampleLen, charSetSize] : undefined,
      })
    );
  }
  model.add(tf.layers.dense({ units: charSetSize, activation: "softmax" }));
  return model;
}

export function compileModel(model: tf.Sequential, learningRate = 0.002) {
  const optimizer = tf.train.rmsprop(learningRate);
  model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });
  console.log(`Compiled model with learning rate ${learningRate}`);
  model.summary();
}

export async function fitModel(
  model: tf.Sequential,
  textData: TextData,
  numEpochs: number,
  examplesPerEpoch = 10000,
  batchSize = 128,
  validationSplit = 0.0625,
  callbacks?: any
) {
  for (let i = 0; i < numEpochs; ++i) {
    const [xs, ys] = textData.nextDataEpoch(examplesPerEpoch);
    await model.fit(xs, ys, {
      epochs: 1,
      batchSize: batchSize,
      validationSplit,
      callbacks,
    });
    xs.dispose();
    ys.dispose();
  }
}

/**
 * Draw a sample based on probabilities.
 *
 * @param {tf.Tensor} probs Predicted probability scores, as a 1D `tf.Tensor` of
 *   shape `[charSetSize]`.
 * @param {tf.Tensor} temperature Temperature (i.e., a measure of randomness
 *   or diversity) to use during sampling. Number be a number > 0, as a Scalar
 *   `tf.Tensor`.
 * @returns {number} The 0-based index for the randomly-drawn sample, in the
 *   range of `[0, charSetSize - 1]`.
 */
export function sample(probs: tf.Tensor, temperature) {
  return tf.tidy(() => {
    const logits: any = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
    const isNormalized = false;
    // `logits` is for a multinomial distribution, scaled by the temperature.
    // We randomly draw a sample from the distribution.
    return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
  });
}

/**
 * Generate text using a next-char-prediction model.
 *
 * @param {tf.Model} model The model object to be used for the text generation,
 *   assumed to have input shape `[null, sampleLen, charSetSize]` and output
 *   shape `[null, charSetSize]`.
 * @param {number[]} sentenceIndices The character indices in the seed sentence.
 * @param {number} length Length of the sentence to generate.
 * @param {number} temperature Temperature value. Must be a number >= 0 and
 *   <= 1.
 * @param {(char: string) => Promise<void>} onTextGenerationChar An optinoal
 *   callback to be invoked each time a character is generated.
 * @returns {string} The generated sentence.
 */
export async function generateText(
  model,
  textData,
  sentenceIndices: number[],
  length,
  temperature,
  onTextGenerationChar?: (char: string) => Promise<void>
) {
  const sampleLen = model.inputs[0].shape[1];
  const charSetSize = model.inputs[0].shape[2];

  // Avoid overwriting the original input.
  sentenceIndices = sentenceIndices.slice();

  let generated = "";
  while (generated.length < length) {
    // Encode the current input sequence as a one-hot Tensor.
    const inputBuffer = tf.buffer([1, sampleLen, charSetSize]);

    // Make the one-hot encoding of the seeding sentence.
    for (let i = 0; i < sampleLen; ++i) {
      inputBuffer.set(1, 0, i, sentenceIndices[i]);
    }
    const input = inputBuffer.toTensor();

    // Call model.predict() to get the probability values of the next
    // character.
    const output = model.predict(input);

    // Sample randomly based on the probability values.
    const winnerIndex = sample(tf.squeeze(output), temperature);
    const winnerChar = textData.getFromCharSet(winnerIndex);
    if (onTextGenerationChar != null) {
      await onTextGenerationChar(winnerChar);
    }

    generated += winnerChar;
    sentenceIndices = sentenceIndices.slice(1);
    sentenceIndices.push(winnerIndex);

    // Memory cleanups.
    input.dispose();
    output.dispose();
  }
  return generated;
}
