import datasets
from llama_recipes.datasets.utils import Concatenator
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


def load_jsonl_dataset(file: str, tokenizer, split):
    full_dataset = datasets.load_dataset(
        "train-data", data_files="train-sample.jsonl", split="train"
    )

    # Since the dataset has no train/test split, we create one and select it
    dataset = full_dataset.train_test_split(
        train_size=0.95,
        test_size=0.05,
        seed=42,
    )["train" if split == "train" else "test"]

    dataset = dataset.map(
        lambda x: tokenizer(x["text"]), remove_columns=list(dataset.features)
    )

    dataset = dataset.map(Concatenator(), batched=True)
    return dataset


def save_hf_model(model_id: str, new_model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=False,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.save_pretrained(new_model_id)
    return model


def load_hf_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_hf_model_for_finetune(model_id: str):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        # device_map={"": 0},
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def load_stf_trainer(model, tokenizer, train_data, formatting_prompts_func):
    peft_args = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True, #False
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    print("Set supervised fine-tuning parameters")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        formatting_func=formatting_prompts_func,
        peft_config=peft_args,
        dataset_text_field="text",
        max_seq_length=1024 * 2,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    return trainer