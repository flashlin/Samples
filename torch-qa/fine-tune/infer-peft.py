import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import transformers
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
from finetune_lit import load_peft_model, ask_llama2_instruction_prompt

config = load_finetune_config()
device = "cuda"


model_name = config['model_name']
base_model = f"./models/{model_name}"
peft_model = f"./outputs"

model, tokenizer = load_peft_model(base_model, peft_model)

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.2  # 0.7
# generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


with torch.inference_mode():
    while True:
        user_input = input("query: ")
        if user_input == '/bye':
            break

        answer = ask_llama2_instruction_prompt(model=model,
                                               generation_config=generation_config,
                                               tokenizer=tokenizer,
                                               device=device,
                                               question=user_input)
        print(answer)
