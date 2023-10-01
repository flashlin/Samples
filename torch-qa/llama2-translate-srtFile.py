import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from srt_file_utils import get_srt_file_metadata_iter, create_srt_line, get_srt_file_metadata, try_parse_time_line
import sys

arguments = sys.argv
sour_srt_file = arguments[1]
dest_srt_file = arguments[2]


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


def find_and_output(s):
    index = s.find('[/INST]')
    if index != -1:
        result = s[index + len('[/INST]'):]
    else:
        result = ''
    result = result.replace('</s>', '')
    return result


def translate_en(text: str) -> str:
    prompt = instruction.format(f"直接翻譯為中文.\r\n{text}")
    generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(),
                                  max_new_tokens=200, streamer=streamer)
    answer = tokenizer.decode(generate_ids[0])
    answer = find_and_output(answer)
    return answer


def read_old_srt_file_count(srt_filepath: str) -> int:
    if not os.path.exists(srt_filepath):
        return 0
    count = 0
    for _ in get_srt_file_metadata(srt_filepath):
        count += 1
    return count


def translate_srt_file(srt_filepath: str, dest_srt_filepath: str):
    old_dest_srt_count = read_old_srt_file_count(dest_srt_filepath)
    total_count = 1
    with open(dest_srt_filepath, "a", encoding='utf-8') as f:
        for count, start_time, end_time, caption in get_srt_file_metadata(srt_filepath):
            if total_count <= old_dest_srt_count:
                total_count += 1
                continue
            print(f"{count} {start_time} {end_time}")
            translated = translate_en(caption)
            line = create_srt_line(start_time, end_time, f"{caption}\r\n{translated}\r\n")
            f.write(f"{total_count}\r\n")
            f.write(line)
            f.flush()
            total_count += 1


translate_srt_file(sour_srt_file, dest_srt_file)
