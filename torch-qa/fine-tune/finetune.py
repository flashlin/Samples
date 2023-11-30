from datasets import load_dataset
import pandas as pd
from finetune_lit import (export_hf_model, load_hf_model_for_finetune,
                          load_hf_tokenizer, load_stf_trainer, save_trainer_model)
from finetune_utils import load_finetune_config

config = load_finetune_config()
model_name = config["model_name"]


def load_train_csv_file(csv_file: str):
    # df = load_dataset(csv_file, split="train")
    df = load_dataset('csv', data_files=csv_file, split="train")
    # df = pd.read_csv(csv_file, header=0)
    print(f"{df.column_names=}")
    # split_point = int(0.1 * len(df))
    # train_data = df.iloc[:split_point]
    # val_data = df.iloc[split_point:]
    # return train_data, val_data
    return df


# base_model = "./models/llama-2-7b-hf"
# new_model = "./models/llama-2-7b-chat-finetuned"
# base_model = "./models/llama-2-13b-chat-hf"
# new_model = "./models/llama-2-13b-chat-finetuned"
# new_model = "./models/FlagAlpha_Atom-7B-Chat"
# #save_hf_model(base_model, new_model)
# #exit(0)

base_model = f"../models/{model_name}"

dataset = load_train_csv_file("./results/train.csv")

print(f"Loading model {model_name}")
model = load_hf_model_for_finetune(base_model)
print(f"{model.config=}")

# tokenizer.json
# tokenizer.model
# tokenizer_config.json
print("Loading tokenizer")
tokenizer = load_hf_tokenizer(base_model)
print("done")


def formatting_prompts_func(example):
    output_texts = []
    # for i in range(len(example['prompt'])):
    #     text = f"### Input: ```{example['prompt'][i]}```\n ### Output: {example['completion'][i]}"
    #     output_texts.append(text)
    for i in range(len(example['text'])):
        text = example['text'][i]
        output_texts.append(text)
    return output_texts


trainer = load_stf_trainer(model, tokenizer, dataset, formatting_prompts_func, config)
print("Start finetune")
trainer.train()
#trainer.train(resume_from_checkpoint=True)
# trainer.train(resume_from_checkpoint="{<path-where-checkpoint-were_stored>/checkpoint-0000")
print("Save model")
save_trainer_model(trainer, config)
print("done")
