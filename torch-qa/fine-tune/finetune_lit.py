import csv
import os.path
import re
from typing import Mapping, Callable, Any
from pathlib import Path
import datasets
from llama_recipes.datasets.utils import Concatenator
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from datasets import Dataset, load_dataset
from io_utils import split_file_path
import logging
import pandas as pd
from torch import float32, nn, exp

# bitsandbytes
# autotrain-advanced 0.6.44
# pip install --upgrade bitsandbytes 0.41.2
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    :param model: PEFT model
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param} || Trainable Parameters: {trainable_params} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )


def load_stf_trainer(model, tokenizer, train_data, formatting_prompts_func, config):
    train_epochs = config['train_epochs']
    if train_epochs is None:
        train_epochs = 10

    train_batch_size = config['train_batch_size']
    if train_batch_size is None:
        train_batch_size = 4

    peft_args = LoraConfig(
        # target_modules=get_last_layer_linears(model),
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,  # 最初的 LoRA 論文建議從 8 級開始，但對於 QLoRA，需要 64 級。
        bias="none",
        task_type="CAUSAL_LM",
    )

    is_QLoRA = config["is_QLoRA"]
    target_modules = config['target_modules'].split(',')
    if not target_modules:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head", ]
    if is_QLoRA:
        peft_args = LoraConfig(
            target_modules=target_modules,
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,  # 最初的 LoRA 論文建議從 8 級開始，但對於 QLoRA，需要 64 級。
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=train_epochs,
        per_device_train_batch_size=train_batch_size,  # 46GB-> 7B:8 13B:4
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=config['save_steps'],
        logging_steps=25,
        learning_rate=1e-4,  # 7B:2e-4 = 0.0002 13B:1e-4 = 0.0001
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    print("trainable Parameters")
    print_trainable_parameters(model, True)

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

    # for name, module in trainer.model.named_modules():
    #     if "norm" in name:
    #         module = module.to(torch.float32)

    return trainer


def get_finetune_model_name(config):
    model_name = config['model_name']
    output_name = f"outputs/{model_name}-peft"
    if config["is_QLoRA"]:
        output_name = f"outputs/{model_name}-qlora"
    return output_name


def save_trainer_model(trainer, config):
    output_name = get_finetune_model_name(config)
    trainer.model.save_pretrained(output_name)
    print(f"save to {output_name}")


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
        # use_safetensors=True
    )
    if peft_model is not None:
        if os.path.exists(peft_model):
            print(f"Loading PEFT {peft_model}")
            model.load_adapter(peft_model)
        else:
            print("WARNING: PEFT_MODEL NOT EXISTS!!!")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def clean_llama2_instruction_resp(resp: str):
    after_inst = resp.split("[/INST]", 1)[-1]
    s2 = after_inst.split("[INST]", 1)[0]
    return s2.split('[/INST]', 1)[0]


def create_llama2_finetune_prompt(question, answer):
    chat_prompt_template = """<s>[INST] {instruction} [/INST] {output}</s>"""
    return chat_prompt_template.format(instruction=question, output=answer)


def create_orca2_finetune_prompt(question, answer):
    template = "You are OpenOrcaChat.<|end_of_turn|>User: {instruction}<|end_of_turn|>Assistant: {output}<|end_of_turn|>"
    return template.format(instruction=question, output=answer)


def create_llama2_generation_prompt(system_message, question: str):
    if system_message is not None:
        return ("<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_input} [/INST]"
                .format(system_message=system_message, user_input=question))
    prompt_template = """<s>[INST] {user_input} [/INST]"""
    return prompt_template.format(user_input=question)


def create_yi_generation_prompt(system_message, question: str):
    return ("<|im_start|>system\n{system_message}<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant").format(system_message=system_message, prompt=question)


def clean_yi_instruction_resp(resp: str):
    after_inst = resp.split("<|im_start|>assistant", 1)[-1]
    s2 = after_inst.split("<|im_start|>system", 1)[0]
    return s2.split('<|im_start|>assistant', 1)[0]


def create_orca2_generation_prompt(system_message, question: str):
    prompt_template = "You are OpenOrcaChat.<|end_of_turn|>User: {instruction}<|end_of_turn|>Assistant: "
    return prompt_template.format(instruction=question)


def ask_llama2_instruction_prompt(model, generation_config, tokenizer, device, question: str):
    system_msg = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.")
    prompt = create_llama2_generation_prompt(system_msg, question)
    encoding = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = clean_llama2_instruction_resp(resp)
    return answer


def ask_yi_instruction_prompt(model, generation_config, tokenizer, device, question: str):
    system_msg = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.")
    prompt = create_yi_generation_prompt(system_msg, question)
    encoding = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=encoding.input_ids,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config,
        do_sample=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=5,
        temperature=0.7,
        top_k=40,
        top_p=0.8,
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = clean_yi_instruction_resp(resp)
    return answer


def ask_orca2_instruction_prompt(model, generation_config, tokenizer, device, question: str):
    prompt = create_orca2_generation_prompt(None, question)
    encoding = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # answer = clean_llama2_instruction_resp(resp)
    answer = resp
    return answer


# orca2_instruction_prompt_template = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
#
# def create_orca2_instruction_prompt(system_message: str, user_input: str) -> str:
#     return orca2_instruction_prompt_template.format(system_message=system_message,
#                                                     prompt=user_input)


def preprocess_text(text: str) -> str:
    text = text.replace('\n', ' ')
    return text


def yield_text_file(text_file_path: Path, tokenizer: PreTrainedTokenizer) -> str:
    """
    Args:
        text_file_path (Path): Extracted text from the book
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer

    Yields:
        str: text of the pages
    """
    with open(text_file_path, 'r') as f:
        grouped_text = ""
        for text in f:
            if len(text.strip()) == 0 and grouped_text != "":
                grouped_text = preprocess_text(grouped_text)
                yield grouped_text + "." + tokenizer.eos_token  # End of paragraphs defined by ".\n is transformed into EOS token"
                grouped_text = ""
            else:
                grouped_text += text


def tokenize(element: Mapping, tokenizer: Callable, context_length: int) -> str:
    inputs = tokenizer(element['text'], truncation=True, return_overflowing_tokens=True,
                       return_length=True, max_length=context_length)
    inputs_batch = []
    for length, input_ids in zip(inputs['length'], inputs['input_ids']):
        if length == context_length:  # We drop the last input_ids that are shorter than max_length
            inputs_batch.append(input_ids)
    return {"input_ids": inputs_batch}


def prepare_dataset(model_name: str, text_file: Path, min_length: int, context_length: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_folder, output_filename, _ = split_file_path(text_file)
    csv_file = f'{output_folder}/{output_filename}.csv'
    LOGGER.info(f'Start preparing dataset from {text_file}')
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['input_ids'])
        for paragraphs in yield_text_file(text_file_path=text_file, tokenizer=tokenizer):
            # input_ids = tokenizer(paragraphs, return_tensors="pt")
            input_ids = tokenizer(paragraphs)['input_ids']
            writer.writerow([input_ids])

    # dataset = Dataset.from_dict({'text': [text]})

    # We push the extracted book publicly
    # dataset.push_to_hub("JeremyArancio/lotr-book")
    # tokenized_dataset = dataset.map(tokenize, batched=True,
    #                                 fn_kwargs={'tokenizer': tokenizer, 'context_length': context_length},
    #                                 remove_columns=dataset.column_names)
    # LOGGER.info(f'The tokenized dataset is composed of {tokenized_dataset.num_rows} elements, each one composed of {context_length} tokens.')
    # tokenized_dataset_dict = tokenized_dataset.train_test_split(test_size=test_size, shuffle=shuffle)
    # tokenized_dataset_dict.push_to_hub(hf_repo)

    # df = tokenized_dataset.to_pandas()
    # with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(df.columns)
    #     writer.writerows(df.values)
    #     print(f"{df.values=}")
    # # df.to_csv(f'{output_folder}/{output_filename}.csv', index=False)
    LOGGER.info(f'Preparing dataset finished.')


# def extract_text_to_jsonl(file: str, jsonl_file: str):
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(float32)


def prepare_model(model):
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(float32)
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    return model

class LLMText():

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = 'cuda'

    def train(
            self,
            dataset_file: str,
            lora_config: Mapping[str, Any],
            trainer_config: Mapping[str, Any],
            mlm: bool,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", load_in_8bit=True)
        model = prepare_model(model)
        model = get_peft_model(model, LoraConfig(**lora_config))
        # LOGGER.info(f"Model trainable parameters:\n {print_trainable_parameters(model)}")
        dataset = load_dataset(dataset_file, streaming=True)
        LOGGER.info(f"Train dataset downloaded:\n {dataset['train']}")
        LOGGER.info(
            f"Number of tokens for the training: {dataset['train'].num_rows * len(dataset['train']['input_ids'][0])}")
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=TrainingArguments(**trainer_config),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
        )
        model.config.use_cache = False  # silence warnings
        trainer.train()
        model.config.use_cache = True

        model.save_pretrained(trainer_config['output_dir'])
        # model.push_to_hub(repo_id=hf_repo)
        # tokenizer.push_to_hub(repo_id=hf_repo)


if __name__ == '__main__':
    model_name = '../models/mistralai_Mistral-7B-Instruct-v0.1'
    # prepare_dataset(model_name,
    #                 text_file='data-user/casino.md',
    #                 min_length=10,
    #                 context_length=2048)

    t = LLMText(model_name=model_name)
    t.train('data-user/casino.csv',
            lora_config={
                'lora_r': 16,
                'lora_alpha': 32,  # alpha scaling
                'lora_dropout': 0.1,
                'lora_bias': "none",
                'lora_task_type': "CAUSAL_LM",
                'target_modules': 'proj_q,proj_v'
            },
            trainer_config={
                'per_device_train_batch_size': 1,
                'gradient_accumulation_steps': 1,
                'warmup_steps': 100,
                'num_train_epochs': 2,
                'weight_decay': 0.1,
                'learning_rate': 1e-4,
                'fp16': True,
                'evaluation_strategy': "no"
            },
            mlm=True)
