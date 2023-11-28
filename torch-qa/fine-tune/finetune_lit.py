import re
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
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import json


# bitsandbytes
# autotrain-advanced 0.6.44
# pip install --upgrade bitsandbytes 0.41.2

def get_num_layers(model):
    numbers = set()
    for name, _ in model.named_parameters():
        for number in re.findall(r'\d+', name):
            numbers.add(int(number))
    return max(numbers)


def get_last_layer_linears(model):
    names = []
    num_layers = get_num_layers(model)
    for name, module in model.named_modules():
        if str(num_layers) in name and not "encoder" in name:
            if isinstance(module, torch.nn.Linear):
                names.append(name)
    return names


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


def export_hf_model(model_id: str, new_model_id: str):
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


def load_stf_trainer(model, tokenizer, train_data, formatting_prompts_func):
    peft_args = LoraConfig(
        target_modules=get_last_layer_linears(model),
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,  # 最初的 LoRA 論文建議從 8 級開始，但對於 QLoRA，需要 64 級。
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
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
        # formatting_func=formatting_prompts_func,
        peft_config=peft_args,
        dataset_text_field="text",
        max_seq_length=1024 * 4,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    return trainer


def save_sft_model(trainer, model):
    model_to_save = trainer.model.module if hasattr(trainer.model,
                                                    'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained("outputs")
    lora_config = LoraConfig.from_pretrained('outputs')
    model = get_peft_model(model, lora_config)
    model.push_to_hub("ashishpatel26/Llama2_Finetuned_Articles_Constitution_3300_Instruction_Set", create_pr=1)


def load_peft_model(base_model: str, peft_model: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        # trust_remote_code=True,
        local_files_only=True,
    )
    if peft_model is not None:
        model.load_adapter(peft_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def clean_llama2_instruction_resp(resp: str):
    after_inst = resp.split("[/INST]", 1)[-1]
    s2 = after_inst.split("[INST]", 1)[0]
    return s2.split('[/INST]', 1)[0]


def create_llama2_generation_prompt(system_message, question: str):
    if system_message is not None:
        return ("<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_input} [/INST]"
                .format(system_message=system_message, user_input=question))
    prompt_template = """<s>[INST] {user_input} [/INST]"""
    return prompt_template.format(user_input=question)


def ask_llama2_instruction_prompt(model, generation_config, tokenizer, device, question: str):
    prompt = create_llama2_generation_prompt(None, question)
    encoding = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = clean_llama2_instruction_resp(resp)
    return answer


orca2_instruction_prompt_template = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"


def create_orca2_instruction_prompt(system_message: str, user_input: str) -> str:
    return orca2_instruction_prompt_template.format(system_message=system_message,
                                                    prompt=user_input)
