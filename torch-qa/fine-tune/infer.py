import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from finetune_utils import load_finetune_config

config = load_finetune_config()


def clean_prompt_resp(resp: str):
    after_inst = resp.split("[/INST]", 1)[-1]
    s2 = after_inst.split("[INST]", 1)[0]
    return s2.split('[/INST]', 1)[0]


model_name = config['model_name']
PEFT_MODEL = f"./models/{model_name}"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    # trust_remote_code=True,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.2  # 0.7
# generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

device = "cuda:0"

prompt = """
<human>: Who brought you into existence?
<assistant>:
""".strip()

prompt_template = """<s>[INST] {user_input} [/INST]"""

with torch.inference_mode():
    while True:
        user_input = input("query: ")
        if user_input == '/bye':
            break

        prompt = prompt_template.format(user_input=user_input)
        encoding = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )

        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = clean_prompt_resp(resp)
        print(answer)