import os
import random
import re

from preprocess_data import TranslationFileTextIterator
from utils.linq_tokenizr import linq_tokenize
from utils.tsql_tokenizr import tsql_tokenize


def replace_many_spaces(text):
    new_text = re.sub(' +', ' ', text)
    return new_text


def filter_tokens(tokens):
    for token in tokens:
        if replace_many_spaces(token.text) != ' ':
            yield token.text


class TranslationTokensIterator:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        for src, tgt in TranslationFileTextIterator(self.file_path):
            src_tokens = linq_tokenize(src)
            tgt_tokens = tsql_tokenize(tgt)
            src = ' '.join(x for x in filter_tokens(src_tokens))
            tgt = ' '.join(x for x in filter_tokens(tgt_tokens))
            yield src, tgt


target_path = r'D:\demo\samples\OpenNMT-py\toy-linq_sql'


def write_train_data(mode, src, tgt):
    with open(f"{target_path}\\src-{mode}.txt", "a+", encoding='UTF-8') as src_f:
        src_f.write(src)
    with open(f"{target_path}\\tgt-{mode}.txt", "a+", encoding='UTF-8') as tgt_f:
        tgt_f.write(tgt)


def remove_file(file_path):
    os.remove(file_path) if os.path.exists(file_path) else None


def write_train_files():
    remove_file(f"{target_path}\\src-train.txt")
    remove_file(f"{target_path}\\tgt-train.txt")
    remove_file(f"{target_path}\\src-val.txt")
    remove_file(f"{target_path}\\tgt-val.txt")
    for src, tgt in TranslationTokensIterator('../data/linq-sample.txt'):
        mode = 'train' if random.randint(1, 10) >= 3 else 'val'
        write_train_data(mode, src, tgt)


if __name__ == '__main__':
    write_train_files()
