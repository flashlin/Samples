import fs from "fs";
import * as os from "os";
import * as path from "path";
import * as tf from "@tensorflow/tfjs";
import { ArgumentParser } from "argparse";
import {
  compileModel,
  fitModel,
  generateText,
  TextData,
} from "@/models/AutoCompleteModel";
import { linqStringToIndexList } from "./linq-encoder";
import { tsqlToIndexList } from "./tsql-encoder";

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

async function train(args) {
  const lstmLayerSize =
    args.lstmLayerSize.indexOf(",") === -1
      ? Number.parseInt(args.lstmLayerSize)
      : args.lstmLayerSize.split(",").map((x: string) => Number.parseInt(x));

  const text = `word`;
  const textData = new TextData(
    "text-data",
    text,
    args.sampleLen,
    args.sampleStep
  );

  const model = createModel(100, textData.charSetSize(), lstmLayerSize);

  compileModel(model);

  const [seed, seedIndices] = textData.getRandomSlice();
  console.log(`Seed text:\n"${seed}"\n`);

  const DISPLAY_TEMPERATURES = [0, 0.25, 0.5, 0.75];
  let epochCount = 0;
  await fitModel(
    model,
    textData,
    args.epochs,
    args.examplesPerEpoch,
    args.batchSize,
    args.validationSplit,
    {
      onTrainBegin: async () => {
        epochCount++;
        console.log(`Epoch ${epochCount} of ${args.epochs}:`);
      },
      onTrainEnd: async () => {
        DISPLAY_TEMPERATURES.forEach(async (temperature) => {
          const generated = await generateText(
            model,
            textData,
            seedIndices,
            args.displayLength,
            temperature
          );
          console.log(
            `Generated text (temperature=${temperature}):\n` +
              `"${generated}"\n`
          );
        });
      },
    }
  );

  console.log("END");

  //await model.save(`file://${args.savePath}`);
  await model.save(`file://./model`);
}

(async () => {
  const args = parseArgs();

  require("@tensorflow/tfjs-node-gpu");
  //require('@tensorflow/tfjs-node');

  // let localTextDataPath = args.textDatasetPath;
  // const text = fs.readFileSync(localTextDataPath, { encoding: "utf-8" });
  prepareTrainData();
})();
