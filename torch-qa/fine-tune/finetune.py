import os
import numpy as np
from datasets import load_dataset
import pandas as pd
from finetune_lit import (export_hf_model, load_hf_model_for_finetune,
                          load_hf_tokenizer, load_stf_trainer, save_trainer_model)
from finetune_utils import load_finetune_config
from torch.utils.data import Dataset

config = load_finetune_config()
model_name = config["model_name"]


def split_train_csv_file(csv_file: str, split_size: int=1000):
    dirname = os.path.dirname(csv_file)
    filename = os.path.basename(csv_file)
    filename_only = os.path.splitext(filename)[0]
    df = pd.read_csv(csv_file)
    num_splits = (len(df) // split_size) + 1
    filenames = []
    for i, chunk in enumerate(np.array_split(df, num_splits)):
        new_filename = f'{dirname}/{filename_only}{i + 1}.csv'
        filenames.append(new_filename)
        chunk.to_csv(new_filename, index=False)
    return filenames


class CustomDataset(Dataset):
    def __init__(self, csv_files):
        self.csv_files = csv_files
        self.len = sum([len(pd.read_csv(csv)) for csv in self.csv_files])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        current_idx = 0
        for i, csv_file in enumerate(self.csv_files):
            df = pd.read_csv(csv_file)
            if current_idx + len(df) > idx:
                selected_row = idx - current_idx
                # return df.iloc[selected_row]
                data_sample = df.iloc[selected_row].to_dict()
                data_sample['input_ids'] = idx
                return data_sample
            current_idx += len(df)
        raise IndexError("Index out of range")



from torch.utils.data import DataLoader
def load_train_csv_file(csv_file: str):
    csv_files = split_train_csv_file(csv_file, 100)
    df = CustomDataset(csv_files)
    # df = load_dataset('csv', data_files=csv_file, split="train")
    # df = load_dataset('csv', data_files=csv_files, split="train")
    df = DataLoader(df, batch_size=2, shuffle=True)
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
