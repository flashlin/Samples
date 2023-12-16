import argparse
import re

from io_utils import split_file_path


def get_args():
    parser = argparse.ArgumentParser(description="Translate Text ")
    parser.add_argument("srt", nargs='?', help="Name of your model")
    args = parser.parse_args()
    return args

NUMBER_PATTERN = re.compile(r'^\d+$')
TIME = '\d+:\d+:\d+\,\d+'
TIME_PATTERN = re.compile(TIME + ' --> ' + TIME)


def translate(text):
    return text


def yield_file_lines(filename):
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
            yield translate(line)


# A_Virtual_World_JavaScript_Course_Lesson_1_11
if __name__ == '__main__':
    args = get_args()
    srt = args.srt
    folder, name, ext = split_file_path(srt)
    dest = f"{folder}/{name}-tw{ext}"
    for line in yield_file_lines(srt):
        print(line)
