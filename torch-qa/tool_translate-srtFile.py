import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from io_utils import split_file_path
from srt_file_utils import get_srt_file_metadata_iter, create_srt_line, get_srt_file_metadata, try_parse_time_line
import sys


# conda install -y -c conda-forge libsndfile

arguments = sys.argv
sour_srt_file = arguments[1]
dest_srt_file = arguments[2]


from seamless_communication.models.inference import Translator

model = "seamlessM4T_large"
model = "multitask_unity_large.pt"
translator = Translator(model,
                        vocoder_name_or_card="vocoder_36langs",
                        device=torch.device("cuda:0"))

# https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/assets/cards/vocoder_36langs.yaml
def translate_en(text: str) -> str:
    src_lang="eng"
    src_lang="jpn"
    tgt_lang="cmn"
    translated_text, _, _ = translator.predict(text, "t2tt", tgt_lang, src_lang=src_lang)
    return translated_text


def read_old_srt_file_count(srt_filepath: str) -> int:
    if not os.path.exists(srt_filepath):
        return 0
    count = 0
    for _ in get_srt_file_metadata(srt_filepath):
        count += 1
    return count


def translate_srt_file_to_zh(srt_filepath: str):
    path, filename, ext = split_file_path(srt_filepath)
    dest_srt_filepath = f"{path}/{filename}.zh.srt"
    old_dest_srt_count = read_old_srt_file_count(dest_srt_filepath)
    total_count = 1
    with open(dest_srt_filepath, "a", encoding='utf-8') as f:
        for count, start_time, end_time, caption in get_srt_file_metadata(srt_filepath):
            if total_count <= old_dest_srt_count:
                total_count += 1
                continue
            caption = caption.strip()
            translated = translate_en(caption)
            #translated = translated.replace('⁇', '')
            print(f"{count} {start_time} {end_time}")
            print(f"{translated}")
            text = f"{caption}\r\n{translated}\r\n".strip()
            line = create_srt_line(start_time, end_time, f"{text}")
            f.write(f"{total_count}\r\n")
            f.write(line)
            f.flush()
            total_count += 1


translate_srt_file_to_zh(sour_srt_file)
