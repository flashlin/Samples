import csv
import os.path
import re
from typing import Mapping, Callable, Any, List
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import Dataset, load_dataset, DatasetDict
import logging
import pandas as pd
from torch import float32, nn, exp
import yaml

from io_utils import query_sub_files, split_file_path

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ['WANDB_MODE'] = 'offline'

def load_finetune_config():
    with open('finetune-text.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_hf_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_hf_model_for_finetune(model_id: str):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": 0},
        local_files_only=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def preprocess_text(text: str) -> str:
    #text = text.replace('\n', ' ')
    return text.strip()


def yield_text_file(text_file_path: Path, tokenizer: PreTrainedTokenizer) -> str:
    with open(text_file_path, 'r') as f:
        grouped_text = ""
        for text in f:
            if len(text.strip()) == 0:
                if grouped_text != "":
                    grouped_text = preprocess_text(grouped_text)
                    yield grouped_text + tokenizer.eos_token
                grouped_text = ""
            else:
                grouped_text += text


def append_to_csv_file(row_data: List[Any], train_file: str):
    with open(train_file, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row_data)


def yield_paragraphs_from_folder(folder: str, tokenizer: PreTrainedTokenizer):
    for text_file in query_sub_files(folder, ['.txt', '.md']):
        for paragraphs in yield_text_file(text_file_path=text_file, tokenizer=tokenizer):
            yield paragraphs

def prepare_dataset(folder: str, output_folder: str, mode: str='w') -> None:
    csv_file = f'{output_folder}/text.csv'
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    LOGGER.info(f'Start preparing dataset from {folder}/')
    with open(csv_file, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['content'])
        for paragraphs in yield_paragraphs_from_folder(folder=folder, tokenizer=tokenizer):
            writer.writerow([paragraphs])
    LOGGER.info(f'Preparing dataset finished.')


class LLMText:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = 'cuda'

    def train(
            self,
            lora_config: Mapping[str, Any],
            trainer_config: Mapping[str, Any],
            mlm: bool,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        context_length = 1024
        def tokenize_context(element):
            outputs = tokenizer(
                element["content"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length <= context_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        # dataset = load_dataset(dataset_file)
        ds_train = load_dataset('csv',
                                data_files={
                                    'train': f'./{output_folder}/text.csv'
                                },
                                column_names=['content'],
                                skiprows=1,
                                split="train")
        
        raw_datasets = DatasetDict(
            {
                "train": ds_train,  # .shuffle().select(range(50000)),
                #"valid": ds_valid,  # .shuffle().select(range(500))
            }
        )

        tokenized_dataset = raw_datasets.map(
            tokenize_context,
            batched=True,
            batch_size=1,
            remove_columns=raw_datasets['train'].column_names
        )

        model = load_hf_model_for_finetune(self.model_name)
        model = get_peft_model(model, LoraConfig(**lora_config))
        # LOGGER.info(f"Model trainable parameters:\n {print_trainable_parameters(model)}")
        # dataset = load_dataset(dataset_file, streaming=False)

        trainer = Trainer(
            model=model,
            train_dataset=tokenized_dataset['train'],
            args=TrainingArguments(**trainer_config),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm),
        )
        model.config.use_cache = False
        trainer.train()
        model.config.use_cache = True
        model.save_pretrained(trainer_config['output_dir'])


if __name__ == '__main__':
    config = load_finetune_config()
    model_name = config['model_name']
    model_name_path = f'../models/{model_name}'
    output_folder = './outputs'

    prepare_dataset(folder='./data-user', output_folder=output_folder)
    prepare_dataset(folder='./data-web', output_folder=output_folder, mode='a')

    t = LLMText(model_name=model_name_path)
    t.train(
            lora_config={
                'r': 64,
                'lora_alpha': 16,  # alpha scaling
                'lora_dropout': 0.1,
                'bias': "none",
                'task_type': "CAUSAL_LM",
                # 'target_modules': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head']
                'target_modules': ['up_proj', 'down_proj', 'lm_head']
            },
            trainer_config={
                'per_device_train_batch_size': 1,
                'gradient_accumulation_steps': 1,
                'warmup_steps': 100,
                'num_train_epochs': config['train_epochs'],
                'weight_decay': 0.1,
                'learning_rate': 1e-4,
                'fp16': False,
                'evaluation_strategy': "no",
                'output_dir': f'./{output_folder}/{model_name}-text',
                'resume_from_checkpoint': ''
                #'max_steps': 160  # (num_samples // batch_size) // gradient_accumulation_steps * epochs
            },
            mlm=False)