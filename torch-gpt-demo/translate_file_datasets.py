from typing import IO

import torch
import torch.utils.data as Data
import itertools

from tsql_tokenizr import tsql_tokenize
from vocabulary_utils import WordVocabulary


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


def pad_words(words: list[str], max_len: int, pad: str) -> list[str]:
    len_words = len(words)
    if len_words < max_len:
        return words + [pad] * (max_len - len_words)
    return words


def pad_zip_words(src_words: list[str], tgt_words: list[str],
                  max_len: int, pad: str) -> list[[str, str]]:
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
        len_range = len_range-max_len+1
    result = []
    for i in range(len_range):
        sub_src = pad_words(src_words[i:i + max_len], max_len, pad)
        sub_tgt = pad_words(tgt_words[i:i + max_len], max_len, pad)
        result.append([sub_src, sub_tgt])
    return result


def read_file_to_csv(file_path: str):
    word_vob = WordVocabulary()
    sos = word_vob.vocab.SOS
    eos = word_vob.vocab.EOS
    pad = word_vob.vocab.PAD

    for line_pair in read_lines_from_file(file_path, n_lines=2):
        src, tgt = tuple(line_pair)
        src_tokens = tsql_tokenize(src)
        tgt_tokens = tsql_tokenize(tgt)
        src_words = [sos] + [token.text for token in src_tokens] + [eos]
        tgt_words = [sos] + [token.text for token in tgt_tokens] + [eos]
        padded_pair_list = pad_zip_words(src_words, tgt_words, max_len=5, pad=pad)
        for padded_src, padded_tgt in padded_pair_list:
            src_index_list = word_vob.encode_many_words(padded_src)
            tgt_index_list = word_vob.encode_many_words(padded_tgt)
            print(f'{padded_src=} {padded_tgt=}')


# # Encoder_input    Decoder_input        Decoder_output
# sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
#              ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
#              ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]  # P: 占位符号，如果当前句子不足固定长度用P占位
#
# src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}  # 词源字典  字：索引
# src_idx2word = {src_vocab[key]: key for key in src_vocab}
# src_vocab_size = len(src_vocab)  # 字典字的个数
# tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
# idx2word = {tgt_vocab[key]: key for key in tgt_vocab}  # 把目标字典转换成 索引：字的形式
# tgt_vocab_size = len(tgt_vocab)  # 目标字典尺寸
# src_len = len(sentences[0][0].split(" "))  # Encoder输入的最大长度
# tgt_len = len(sentences[0][1].split(" "))  # Decoder输入输出最大长度
#
# vocab = WordVocabulary()
#
# file = './data/tsql.txt'
# with open(file, 'r', encoding='utf-8') as f:
#     line = f.readline()
#     while line:
#         src = line
#         tgt = line = f.readline()
#
#
# # 把sentences 转换成字典索引
# def make_data():
#     enc_inputs, dec_inputs, dec_outputs = [], [], []
#
#     for i in range(len(sentences)):
#         enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
#         dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
#         dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
#         enc_inputs.extend(enc_input)
#         dec_inputs.extend(dec_input)
#         dec_outputs.extend(dec_output)
#     return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
#
#
# # 自定义数据集函数
# class MyDataSet(Data.Dataset):
#     def __init__(self, enc_inputs, dec_inputs, dec_outputs):
#         super(MyDataSet, self).__init__()
#         self.enc_inputs = enc_inputs
#         self.dec_inputs = dec_inputs
#         self.dec_outputs = dec_outputs
#
#     def __len__(self):
#         return self.enc_inputs.shape[0]
#
#     def __getitem__(self, idx):
#         return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

if __name__ == '__main__':
    translate_file_path = './data/tsql.txt'
    read_file_to_csv(translate_file_path)
