#$ autotrain llm -h
#$ pip install -q peft  accelerate bitsandbytes safetensors

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
model_name = "bn22/Mistral-7B-Instruct-v0.1-sharded" #"mistralai/Mistral-7B-Instruct-v0.1"
adapters_name = "ashishpatel26/mistral-7b-mj-finetuned"


device = "cuda" # the device to load the model onto
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map='auto'
)
model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token_id = 1

text = "[INST] generate a midjourney prompt for A person walks in the rain [/INST]"

encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
model_input = encoded
model.to(device)
generated_ids = model.generate(**model_input, max_new_tokens=2000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
