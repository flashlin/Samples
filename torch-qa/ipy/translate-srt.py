import argparse
import re
import torch
from io_utils import split_file_path
from seamless_communication.inference import Translator


def get_args():
    parser = argparse.ArgumentParser(description="Translate Text ")
    parser.add_argument("srt", nargs='?', help="Name of your model")
    args = parser.parse_args()
    return args

NUMBER_PATTERN = re.compile(r'^\d+$')
TIME = '\d+:\d+:\d+\,\d+'
TIME_PATTERN = re.compile(TIME + ' --> ' + TIME)


def yield_file_lines(filename, translator):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if NUMBER_PATTERN.match(line):
                yield line
                continue
            if TIME_PATTERN.match(line):
                yield line
                continue
            if line == "":
                yield line
                continue
            yield line
            yield translator.translate(line)


class MyTranslator:
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        self.translator = Translator(
            model_name_or_card="seamlessM4T_v2_large_local",
            vocoder_name_or_card="vocoder_v2_local",
            device=device,
            dtype=dtype,
            apply_mintox=True,
        )
    def translate(self, text):
        translated_text, _ = self.translator.predict(text, "t2tt", 'cmn', src_lang='eng')
        output = f"{translated_text[0]}"
        output = output.replace("⁇", "")
        return output


#python ./translate-srt.py /mnt/d/Demo/AI-mp4/A_Virtual_World_JavaScript_Course_Lesson_1_11.srt
if __name__ == '__main__':
    args = get_args()
    srt = args.srt
    folder, name, ext = split_file_path(srt)
    dest_srt = f"{folder}-processed/{name}{ext}"
    translator = MyTranslator()

    with open(dest_srt, 'w', encoding='utf') as f:
        for line in yield_file_lines(srt, translator):
            f.write(f"{line}\r\n")
            print(line)

