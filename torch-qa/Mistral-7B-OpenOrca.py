import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import textwrap

# pip install --upgrade git+https://github.com/huggingface/transformers


MODEL_PATH = "models/Open-Orca_Mistral-7B-OpenOrca"
device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                          torch_dtype="auto")

print("loading model")
# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
# 12GB GPU: Load failed 2023-10-04
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                             torch_dtype="auto",
                                             #quantization_config=quantization_config,
                                             )

text = """<|im_start|>system\n
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!\n
<|im_end|>\n
<|im_start|>user\n
what is the meaning of life?\n
<|im_end|>"""

encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False)
model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


def wrap_text(text, width=90):  # preserve_newlines
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


def generate(input_text, system_prompt="", max_length=512):
    if system_prompt != "":
        system_prompt = f"""<|im_start|> system\n{system_prompt}<|im_end|>"""
    else:
        system_prompt = ""
    prompt = f"""<|im_start|> user\n{input_text}<|im_end|>"""
    final_prompt = system_prompt + prompt
    inputs = tokenizer(final_prompt, return_tensors="pt", add_special_tokens=False)
    model_inputs = encodeds.to(device)
    model.to(device)
    outputs = model.generate(**inputs,
                             max_length=max_length,
                             temperature=0.1,
                             pad_token_id=3200,
                             do_sample=True)
    text = tokenizer.batch_decode(outputs)[0]
    # text = text[len(final_prompt):] if text.startswith(final_prompt) else text
    text = text.replace(final_prompt, '', 1)
    wrapped_text = wrap_text(text)
    print(wrapped_text)


generate('''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """''', system_prompt="You are a genius python coder")
