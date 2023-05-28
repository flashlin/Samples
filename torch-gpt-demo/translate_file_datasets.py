from typing import IO
import csv
import torch
import torch.utils.data as Data
import itertools
from enum import Enum
from typing import TypeVar, List
import pandas as pd
from data_utils import write_dict_to_file
from tsql_tokenizr import tsql_tokenize
from vocabulary_utils import WordVocabulary
import ast


def read_lines_from_file_ptr(file_ptr: IO, n_lines: int):
    while True:
        lines = list(itertools.islice(file_ptr, n_lines))
        if not lines:
            break
        lines = [line.rstrip('\n') for line in lines]
        yield lines


def read_lines_from_file(file_path: str, n_lines: int = 2):
    with open(file_path, 'r', encoding='utf-8') as sr:
        line_pairs = read_lines_from_file_ptr(sr, n_lines)
        for lines in line_pairs:
            yield lines


T = TypeVar('T')


def pad_list(value_list: list[T], max_len: int, pad: T) -> list[T]:
    len_values = len(value_list)
    if len_values < max_len:
        return value_list + [pad] * (max_len - len_values)
    return value_list


def pad_zip(src_words: list[T], tgt_words: list[T],
            max_len: int, pad: T) -> list[[T, T, T]]:
    """
    :param src_words: ['a', 'b', 'c']
    :param tgt_words: ['a', 'b', 'c']
    :param max_len: 2
    :param pad: '<PAD>'
    :return: [['a','b'],['b','c']]
    """
    len_range = max(len(src_words), len(tgt_words))
    if len_range < max_len:
        len_range = 1
    if len_range > max_len:
        len_range = len_range - max_len + 1
    result = []
    for i in range(len_range):
        sub_src_words = src_words[i:i + max_len]
        sub_src = pad_list(sub_src_words, max_len, pad)
        sub_tgt1 = pad_list(tgt_words[i:i + max_len], max_len, pad)
        sub_tgt2 = pad_list(tgt_words[i + 1:i + 1 + max_len], max_len, pad)
        result.append([sub_src, sub_tgt1, sub_tgt2])
    return result


def remove_enum(value_list):
    return [item.value if isinstance(item, Enum) else item for item in value_list]


def read_file_to_csv(file_path: str, output_csv_path: str):
    word_vob = WordVocabulary()
    sos_index = word_vob.vocab.SOS_index
    eos_index = word_vob.vocab.EOS_index
    pad_index = word_vob.vocab.PAD_index

    vocab_path = './data/vocab.txt'

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['src_input', 'tgt_input', 'tgt_output', 'encoder_input', 'decoder_input', 'decoder_output'])

        for line_pair in read_lines_from_file(file_path, n_lines=2):
            src, tgt = tuple(line_pair)
            src_tokens = tsql_tokenize(src)
            tgt_tokens = tsql_tokenize(tgt)
            src_words = [token.text for token in src_tokens]
            tgt_words = [token.text for token in tgt_tokens]

            src_values = [sos_index] + word_vob.encode_many_words(src_words) + [eos_index]
            tgt_values = [sos_index] + word_vob.encode_many_words(tgt_words) + [eos_index]

            padded_pair_list = pad_zip(src_values, tgt_values, max_len=900, pad=pad_index)
            for padded_src_values, padded_tgt1_values, padded_tgt2_values in padded_pair_list:
                padded_src = word_vob.decode_value_list(padded_src_values, isShow=True)
                padded_tgt1 = word_vob.decode_value_list(padded_tgt1_values, isShow=True)
                padded_tgt2 = word_vob.decode_value_list(padded_tgt2_values, isShow=True)
                encoder_input = remove_enum(padded_src_values)
                decoder_input = remove_enum(padded_tgt1_values)
                decoder_output = remove_enum(padded_tgt2_values)
                writer.writerow([padded_src, padded_tgt1, padded_tgt2,
                                 encoder_input, decoder_input, decoder_output])
    write_dict_to_file(word_vob.to_serializable(), vocab_path)


class MyDataSet(Data.Dataset):
    def __init__(self, csv_file_path, chunk_size=1):
        super(MyDataSet, self).__init__()
        self.csv_file_path = csv_file_path
        self.chunk_size = chunk_size
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            self.headers = file.readline().strip().split(',')

    def __len__(self):
        with open(self.csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            return sum(1 for _ in csv_reader) - 1  # 減去 header 的行數

    def __getitem__(self, idx):
        data_frame = pd.read_csv(self.csv_file_path, skiprows=idx, nrows=self.chunk_size)
        dict_value = data_frame.iloc[0]
        new_dict = {}
        for index, header in enumerate(self.headers):
            new_dict[header] = dict_value[index]
        return new_dict


def collate_fn(batch):
    processed_batch = []
    for item in batch:
        encoder_input = ast.literal_eval(item["encoder_input"])
        decoder_input = ast.literal_eval(item['decoder_input'])
        decoder_output = ast.literal_eval(item['decoder_output'])
        data = {
            'encoder_input': torch.LongTensor(encoder_input),
            'decoder_input': torch.LongTensor(decoder_input),
            'decoder_output': torch.LongTensor(decoder_output)
        }
        processed_batch.append(data)
    return processed_batch


if __name__ == '__main__':
    translate_file_path = './data/tsql.txt'
    csv_file_path = './data/tsql.csv'
    read_file_to_csv(translate_file_path, csv_file_path)
    data_set = MyDataSet(csv_file_path)
    data_loader = Data.DataLoader(data_set, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for batch in data_loader:
        print(f'{batch=}')
