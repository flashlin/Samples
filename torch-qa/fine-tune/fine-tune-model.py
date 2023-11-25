import pandas as pd
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import json
from finetune_lit import export_hf_model, load_hf_model_for_finetune, load_hf_tokenizer

def convert_parquet_to_csv(parquet_file: str):
    df = pd.read_parquet('train-data/train-sample.parquet')
    df.to_csv('train-data/train-sample.csv')

def convert_csv_to_json(csv_file: str):
    df = pd.read_csv(csv_file, header=0)
    print(f"{df.columns.tolist()=}")
    text_array = df['text'].tolist()
    with open(f'train-sample.json', 'w') as json_file:
        json.dump(text_array, json_file)
    

def load_parquet_file(file: str):
    df = pd.read_parquet(file)    
    column_names = ['id', 'text']
    df.columns = column_names
    return df
    # full_dataset = load_dataset(file, split="train")
    # train_dataset = full_dataset.train_test_split(test_size=0.1)["train"]
    # return train_dataset


def load_train_csv_file(csv_file: str):
    # dataset = load_dataset(guanaco_dataset, split="train")
    df = pd.read_csv(csv_file, header=1)
    split_point = int(0.1 * len(df))
    train_data = df.iloc[:split_point]
    val_data = df.iloc[split_point:]
    return train_data, val_data


def formatting_prompts_func(example):
    output_texts = []
    # for i in range(len(example['prompt'])):
    #     text = f"### Input: ```{example['prompt'][i]}```\n ### Output: {example['completion'][i]}"
    #     output_texts.append(text)
    for i in range(len(example['prompt'])):
        text = example['prompt'][i]
        output_texts.append(text)
    return output_texts


# dataset = open_instruct_dataset.filter(lambda example: (len(example["input"]) + len(example["output"]) + len(example["instruction"])) <= 4096)
# total_data_points = len(dataset)
# sample_size = 5_000
# random_indices = random.sample(range(total_data_points), sample_size)
# subset = dataset.select(random_indices)

# [
#   {
#     "prompt": "she no went to market",
#     "completion": "She didn't go to the market."
#   }
# ]
def load_train_json_file(file: str):
    dataset = load_dataset("json", 
                           data_files=file,
                           split="train")
    return dataset

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"
# Fine-tuned model
new_model = "my-llama-2-7b-chat"

# print("save model")
# save_hf_model(base_model, new_model)

model = load_hf_model_for_finetune(new_model)
print(f"{model.config=}")

print("# Load LLaMA tokenizer")
tokenizer = load_hf_tokenizer(new_model)

# print("# Load LoRA configuration")
# peft_args = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
#
# print("Set training parameters")
# training_params = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=100,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=1,
#     optim="paged_adamw_32bit",
#     save_steps=25,
#     logging_steps=25,
#     learning_rate=2e-4,
#     weight_decay=0.001,
#     fp16=True, #False
#     bf16=False,
#     max_grad_norm=0.3,
#     max_steps=-1,
#     warmup_ratio=0.03,
#     group_by_length=True,
#     lr_scheduler_type="constant",
#     report_to="tensorboard"
# )
#
#
# print("Set supervised fine-tuning parameters")
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     formatting_func=formatting_prompts_func,
#     peft_config=peft_args,
#     # dataset_text_field="text",
#     max_seq_length=1024 * 2,
#     tokenizer=tokenizer,
#     args=training_params,
#     packing=False,
# )
#
# print("Train model...")
# # Train model
# trainer.train()
#
# print("# Save model")
# trainer.model.save_pretrained(new_model)
#
# # not work
# # print("# Run text generation pipeline with our next model")
# # prompt = "Who brought you into existence?"
# # pipe = pipeline(task="text-generation", model=new_model, tokenizer=new_model, max_length=200)
# # result = pipe(f"<s>[INST] {prompt} [/INST]")
# # answer = result[0]['generated_text']
# # print(f"{answer=}")
#
# print("END")
# def push_model():
#     # Reload model in FP16 and merge it with LoRA weights
#     load_model = AutoModelForCausalLM.from_pretrained(
#         base_model,
#         low_cpu_mem_usage=True,
#         return_dict=True,
#         torch_dtype=torch.float16,
#         device_map={"": 0},
#     )
#
#     model = PeftModel.from_pretrained(load_model, new_model)
#     model = model.merge_and_unload()
#
#     # Reload tokenizer to save it
#     tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     model.push_to_hub(new_model, use_temp_dir=False)
#     tokenizer.push_to_hub(new_model, use_temp_dir=False)
#
