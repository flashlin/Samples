import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = "./models/Chinese-Llama-2-7b-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
print("tokenizer loaded")
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             torch_dtype=torch.float16,
                                             device_map='auto'
                                             )
print("model loaded")

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible, while being safe.  
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, 
explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

# prompt = instruction.format("用中文回答，When is the best time to visit Beijing, and do you have any suggestions for me?")
prompt = instruction.format(
    "''When is the best time to visit Beijing, and do you have any suggestions for me?'' 翻譯為中文")

print("starting")
generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(),
                              max_new_tokens=4096, streamer=streamer)

print(f"{generate_ids=}")
answer = tokenizer.decode(generate_ids[0])

def find_and_output(s):
    index = s.find('[/INST]')
    if index != -1:
        result = s[index + len('[/INST]'):]
    else:
        result = ''
    result = result.replace('</s>', '')
    return result


answer = find_and_output(answer)
print(f"{answer=}")

