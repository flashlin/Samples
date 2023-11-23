from datasets import load_dataset
import pandas as pd

from finetune_lit import save_hf_model, load_hf_model_for_finetune, load_hf_tokenizer, load_stf_trainer

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

base_model = "./models/llama-2-7b-hf"
new_model = "./models/llama-2-7b-chat-finetuned"

base_model = "./models/llama-2-13b-chat-hf"
new_model = "./models/llama-2-13b-chat-finetuned"

new_model = "./models/FlagAlpha_Atom-7B-Chat"
# new_model = "./models/atom-7b-chat-finetuned"

#save_hf_model(base_model, new_model)
#exit(0)


dataset = load_train_csv_file("./train.csv")

print("Loading model")
model = load_hf_model_for_finetune(new_model)
print(f"{model.config=}")

# tokenizer.json
# tokenizer.model
# tokenizer_config.json
print("Loading tokenizer")
tokenizer = load_hf_tokenizer(new_model)
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


trainer = load_stf_trainer(model, tokenizer, dataset, formatting_prompts_func)
print("Start finetune")
trainer.train()
print("Save model")
trainer.model.save_pretrained(new_model)
print("done")
