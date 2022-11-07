import * as fs from "fs";
//import * as os from "os";
import * as path from "path";
import * as tf from "@tensorflow/tfjs";
import { ArgumentParser } from "argparse";
import { compileModel } from "@/models/AutoCompleteModel";
import { linqCharacters, linqStringToIndexList } from "./linq-encoder";
import { tsqlCharacters, tsqlToIndexList } from "./tsql-encoder";

function createModel(
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

function seq2seqModel(
  numEncoderTokens: number,
  numDecoderTokens: number,
  latentDim: number
) {
  // Define an input sequence and process it.
  const encoderInputs = tf.layers.input({
    shape: [null, numEncoderTokens] as number[],
    name: "encoderInputs",
  });

  const encoder = tf.layers.lstm({
    units: latentDim,
    returnState: true,
    name: "encoderLstm",
  });
  const [, stateH, stateC] = encoder.apply(
    encoderInputs
  ) as tf.SymbolicTensor[];
  // We discard `encoder_outputs` and only keep the states.
  const encoderStates = [stateH, stateC];

  // Set up the decoder, using `encoder_states` as initial state.
  const decoderInputs = tf.layers.input({
    shape: [null, numDecoderTokens] as number[],
    name: "decoderInputs",
  });
  // We set up our decoder to return full output sequences,
  // and to return internal states as well. We don't use the
  // return states in the training model, but we will use them in inference.
  const decoderLstm = tf.layers.lstm({
    units: latentDim,
    returnSequences: true,
    returnState: true,
    name: "decoderLstm",
  });

  const [decoderOutputs] = decoderLstm.apply([
    decoderInputs,
    ...encoderStates,
  ]) as tf.Tensor[];

  const decoderDense = tf.layers.dense({
    units: numDecoderTokens,
    activation: "softmax",
    name: "decoderDense",
  });

  const decoderDenseOutputs = decoderDense.apply(
    decoderOutputs
  ) as tf.SymbolicTensor;

  // Define the model that will turn
  // `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  const model = tf.model({
    inputs: [encoderInputs, decoderInputs],
    outputs: decoderDenseOutputs,
    name: "seq2seqModel",
  });
  return {
    encoderInputs,
    encoderStates,
    decoderInputs,
    decoderLstm,
    decoderDense,
    model,
  };
}

// function nextDataEpoch(
//   textLength,
//   sampleLen,
//   sampleStep,
//   charSetSize,
//   textIndices,
//   numExamples
// ) {
//   const trainingIndices = [];

//   for (let i = 0; i < textLength - sampleLen - 1; i += sampleStep) {
//     trainingIndices.push(i);
//   }

//   tf.util.shuffle(trainingIndices);

//   const xsBuffer = new tf.TensorBuffer([numExamples, sampleLen, charSetSize]);

//   const ysBuffer = new tf.TensorBuffer([numExamples, charSetSize]);

//   for (let i = 0; i < numExamples; ++i) {
//     const beginIndex = trainingIndices[i % trainingIndices.length];
//     for (let j = 0; j < sampleLen; ++j) {
//       xsBuffer.set(1, i, j, textIndices[beginIndex + j]);
//     }
//     ysBuffer.set(1, i, textIndices[beginIndex + sampleLen]);
//   }

//   return [xsBuffer.toTensor(), ysBuffer.toTensor()];
// }

function parseArgs() {
  const parser = new ArgumentParser({
    description: "Train an lstm-text-generation model.",
  });
  // parser.add_argument('textDatasetNameOrPath', {
  //   type: 'string',
  //   help: 'Name of the text dataset (one of ' +
  //     Object.keys(TEXT_DATA_URLS).join(', ') +
  //     ') or the path to a text file containing a custom dataset'
  // });
  parser.add_argument("--gpu", {
    action: "store_true",
    help: "Use CUDA GPU for training.",
  });
  parser.add_argument("--data", {
    action: "store_true",
    help: "process train raw data",
  });
  parser.add_argument("--sampleLen", {
    type: "int",
    default: 60,
    help:
      "Sample length: Length of each input sequence to the model, in " +
      "number of characters.",
  });
  parser.add_argument("--sampleStep", {
    type: "int",
    default: 3,
    help:
      "Step length: how many characters to skip between one example " +
      "extracted from the text data to the next.",
  });
  parser.add_argument("--learningRate", {
    type: "float",
    default: 1e-2,
    help: "Learning rate to be used during training",
  });
  parser.add_argument("--epochs", {
    type: "int",
    default: 150,
    help: "Number of training epochs",
  });
  parser.add_argument("--examplesPerEpoch", {
    type: "int",
    default: 10000,
    help: "Number of examples to sample from the text in each training epoch.",
  });
  parser.add_argument("--batchSize", {
    type: "int",
    default: 128,
    help: "Batch size for training.",
  });
  parser.add_argument("--validationSplit", {
    type: "float",
    default: 0.0625,
    help: "Validation split for training.",
  });
  parser.add_argument("--displayLength", {
    type: "int",
    default: 120,
    help: "Length of the sampled text to display after each epoch of training.",
  });
  parser.add_argument("--savePath", {
    type: "str",
    help: "Path to which the model will be saved (optional)",
  });
  parser.add_argument("--lstmLayerSize", {
    type: "str",
    default: "128,128",
    help:
      "LSTM layer size. Can be a single number or an array of numbers " +
      'separated by commas (E.g., "256", "256,128")',
  });
  const args = parser.parse_args();
  return args;
}

function prepareTrainData() {
  const outputTrainFile = "./dist/train.txt";
  fs.writeFileSync(outputTrainFile, "");

  const text = fs.readFileSync("./data/linq-sample.txt", "utf8");
  text.split("\n").forEach((line, idx) => {
    if (idx % 2 == 0) {
      console.info(line);
      const values = linqStringToIndexList(line);
      fs.appendFileSync(outputTrainFile, JSON.stringify(values) + "\n");
      return;
    }

    {
      const values = tsqlToIndexList(line);
      fs.appendFileSync(outputTrainFile, JSON.stringify(values) + "\n");
    }
  });
}

async function startTrain(args) {
  const inputTexts: number[][] = [];
  const targetTexts: number[][] = [];

  const dataFile = "./dist/train.txt";
  const inputLines = fs.readFileSync(dataFile, "utf-8");
  inputLines.split("\n").forEach((line, idx) => {
    if (line == "") {
      return;
    }
    if (idx % 2 == 0) {
      const values = JSON.parse(line);
      inputTexts.push(values);
      return;
    }
    {
      const values = JSON.parse(line);
      targetTexts.push(values);
    }
  });

  const numEncoderTokens = linqCharacters.length;
  const numDecoderTokens = tsqlCharacters.length;

  const maxEncoderSeqLength = inputTexts
    .map((text) => text.length)
    .reduceRight((prev, curr) => (curr > prev ? curr : prev), 0);
  const maxDecoderSeqLength = targetTexts
    .map((text) => text.length)
    .reduceRight((prev, curr) => (curr > prev ? curr : prev), 0);

  console.log("Number of samples:", inputTexts.length);
  console.log("Number of unique input tokens:", numEncoderTokens);
  console.log("Number of unique output tokens:", numDecoderTokens);
  console.log("Max sequence length for inputs:", maxEncoderSeqLength);
  console.log("Max sequence length for outputs:", maxDecoderSeqLength);

  const metadata = {
    //input_token_index: inputTokenIndex,
    //target_token_index: targetTokenIndex,
    max_encoder_seq_length: maxEncoderSeqLength,
    max_decoder_seq_length: maxDecoderSeqLength,
  };

  // Save the token indices to file.
  const artifacts_dir = "./dist";
  const metadataJsonPath = path.join(artifacts_dir, "metadata.json");
  fs.writeFileSync(metadataJsonPath, JSON.stringify(metadata));
  console.log("Saved metadata at: ", metadataJsonPath);

  const encoderInputDataBuf = tf.buffer<tf.Rank.R3>([
    inputTexts.length,
    maxEncoderSeqLength,
    numEncoderTokens,
  ]);
  const decoderInputDataBuf = tf.buffer<tf.Rank.R3>([
    inputTexts.length,
    maxDecoderSeqLength,
    numDecoderTokens,
  ]);
  const decoderTargetDataBuf = tf.buffer<tf.Rank.R3>([
    inputTexts.length,
    maxDecoderSeqLength,
    numDecoderTokens,
  ]);

  inputTexts.forEach((inputText, i) => {
    for (const [t, char] of inputText.entries()) {
      encoderInputDataBuf.set(1, i, t, char);
    }
  });

  targetTexts.forEach((targetText, i) => {
    for (const [t, char] of targetText.entries()) {
      decoderInputDataBuf.set(1, i, t, char);
      if (t > 0) {
        // decoder_target_data will be ahead by one timestep
        // and will not include the start character.
        decoderTargetDataBuf.set(1, i, t - 1, char);
      }
    }
  });

  const encoderInputData = encoderInputDataBuf.toTensor();
  const decoderInputData = decoderInputDataBuf.toTensor();
  const decoderTargetData = decoderTargetDataBuf.toTensor();

  const latentDim = 256;
  const {
    encoderInputs,
    encoderStates,
    decoderInputs,
    decoderLstm,
    decoderDense,
    model,
  } = seq2seqModel(numEncoderTokens, numDecoderTokens, latentDim);

  model.compile({
    optimizer: "rmsprop",
    loss: "categoricalCrossentropy",
  });
  model.summary();

  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const tfn = require("@tensorflow/tfjs-node-gpu");

  const model1 = await tf.loadLayersModel("file://./dist/model/model.json");
  model1.compile({
    optimizer: "rmsprop",
    loss: "categoricalCrossentropy",
  });

  const m = model1;

  await m.fit([encoderInputData, decoderInputData], decoderTargetData, {
    batchSize: 32,
    epochs: 10,
    validationSplit: 0.2,
    callbacks:
      args.logDir == null
        ? null
        : tfn.node.tensorBoard(args.logDir, {
            updateFreq: args.logUpdateFreq,
          }),
  });

  console.log("END");

  await m.save(`file://./dist/model`);

  // Define sampling models
  const encoderModel = tf.model({
    inputs: encoderInputs,
    outputs: encoderStates,
    name: "encoderModel",
  });

  const decoderStateInputH = tf.layers.input({
    shape: [latentDim],
    name: "decoderStateInputHidden",
  });

  const decoderStateInputC = tf.layers.input({
    shape: [latentDim],
    name: "decoderStateInputCell",
  });

  const decoderStatesInputs = [decoderStateInputH, decoderStateInputC];
  let [decoderOutputs, stateH, stateC] = decoderLstm.apply([
    decoderInputs,
    ...decoderStatesInputs,
  ]) as tf.SymbolicTensor[];

  const decoderStates = [stateH, stateC];
  decoderOutputs = decoderDense.apply(decoderOutputs) as tf.SymbolicTensor;
  const decoderModel = tf.model({
    inputs: [decoderInputs, ...decoderStatesInputs],
    outputs: [decoderOutputs, ...decoderStates],
    name: "decoderModel",
  });
}

(async () => {
  const args = parseArgs();

  //require("@tensorflow/tfjs-node-gpu");
  //require('@tensorflow/tfjs-node');

  console.log(args);

  if (args.data) {
    console.log("process traing raw data");
    prepareTrainData();
    return;
  }

  startTrain(args);
})();
