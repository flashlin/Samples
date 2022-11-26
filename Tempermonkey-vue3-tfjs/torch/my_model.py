import os
import random
import re
import random
import string

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader

from common.io import info
from preprocess_data import TranslationFileTextIterator
from utils.data_ex import df_to_values, pad_array
from utils.linq_tokenizr import linq_tokenize
from utils.stream import StreamTokenIterator, read_double_quote_string, read_until, int_list_to_str
from utils.template_utils import TemplateText
from utils.tokenizr import create_char2index_map, create_index2char_map
from utils.tsql_tokenizr import tsql_tokenize


def replace_many_spaces(text):
    new_text = re.sub(' +', ' ', text)
    return new_text


def get_data_file_path(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ml/data/{file_name}")


def line_to_tokens(line):
    stream_iter = StreamTokenIterator(line)
    buff = []
    while not stream_iter.is_done():
        ch = stream_iter.peek_str(1)
        if ch == '"':
            buff.append(read_double_quote_string(stream_iter).text)
            continue
        if ch == ' ':
            buff.append(stream_iter.next().text)
            continue
        text = read_until(stream_iter, ' ').text
        buff.append(text)
    return buff


def read_examples(example_file):
    def filter_space_tokens(a_tokens):
        for token in a_tokens:
            if token == ' ':
                continue
            yield token.rstrip()

    with open(example_file, "r", encoding='UTF-8') as f:
        for line in f:
            tokens = line_to_tokens(line)
            tokens = [t for t in filter_space_tokens(tokens)]
            yield tokens


def read_examples_to_tokens3(example_file):
    for idx, tokens in enumerate(read_examples(example_file)):
        if idx % 3 == 0:
            src_tokens = tokens
            continue
        if idx % 3 == 1:
            tgt1_tokens = tokens
            continue
        tgt2_tokens = tokens
        yield src_tokens, tgt1_tokens, tgt2_tokens


def split_space(line):
    return [x.rstrip() for x in line.split(' ')]


def get_vocabs():
    vocab_file = get_data_file_path("linq_classification_vocab.txt")
    with open(vocab_file, "r", encoding='UTF-8') as f:
        lines = f.readlines()
        common_symbols = split_space(lines[0])
        src_tokens = split_space(lines[1])
        tgt_tokens = split_space(lines[2])
    return common_symbols + src_tokens, common_symbols + tgt_tokens


src_symbols, tgt_symbols = get_vocabs()
src_char2index = create_char2index_map(src_symbols)
src_index2char = create_index2char_map(src_symbols)
tgt_char2index = create_char2index_map(tgt_symbols)
tgt_index2char = create_index2char_map(tgt_symbols)


def encode(tokens, char2index):
    var_re = re.compile(r'(@\w.+)(\d+)')
    buff = []
    unk_tokens = {}
    for token in tokens:
        match = var_re.match(token)
        if match:
            name = match.group(1)
            num = match.group(2)
            buff.append(char2index[name])
            buff.append(char2index[num])
            continue
        if token not in char2index:
            unk_num = len(unk_tokens) + 1
            if token in unk_tokens:
                unk = unk_tokens[token]
            else:
                unk = [char2index['<unk>'], char2index[str(unk_num)]]
                unk_tokens[token] = unk
            buff.extend(unk)
            continue
        buff.append(char2index[token])
    return buff


def decode_to_text(values, index2char):
    buff = []
    for value in values:
        buff.append(index2char[value])
    return buff


def encode_src(text):
    return encode(text, src_char2index)


def encode_tgt(text):
    return encode(text, tgt_char2index)


def decode_src_to_text(text):
    return decode_to_text(text, src_index2char)


def decode_tgt_to_text(text):
    return decode_to_text(text, tgt_index2char)


def write_train_files(target_path="./output"):
    def write_train_data():
        f.write(int_list_to_str(src_values))
        f.write('\t')
        f.write(int_list_to_str(tgt1_values))
        f.write('\t')
        f.write(int_list_to_str(tgt2_values))

    remove_file(f"{target_path}\\linq_sql.csv")
    file = get_data_file_path("linq_classification.txt")
    with open(f"{target_path}\\linq_sql.csv", "w", encoding='UTF-8') as f:
        f.write("src\ttgt1\ttgt2")
        for (src, tgt1, tgt2) in read_examples_to_tokens3(file):
            src_values = encode_src(src)
            tgt1_values = encode_src(tgt1)
            tgt2_values = encode_tgt(tgt2)
            write_train_data(src_values, tgt1_values, tgt2_values)


def pad_data_loader(dataset, batch_size, padding_idx, **kwargs):
    def pad_collate(batch):
        (src_input, dec_input, dec_output) = zip(*batch)
        enc_input_pad = torch.nn.utils.rnn.pad_sequence(src_input, batch_first=True, padding_value=padding_idx)
        dec_input_pad = torch.nn.utils.rnn.pad_sequence(dec_input, batch_first=True, padding_value=padding_idx)
        dec_output_pad = torch.nn.utils.rnn.pad_sequence(dec_output, batch_first=True, padding_value=padding_idx)
        return enc_input_pad, dec_input_pad, dec_output_pad

    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=pad_collate, **kwargs)


class TranslationDataset(Dataset):
    def __init__(self, csv_file_path, padding_idx):
        self.padding_idx = padding_idx
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        self.src = df_to_values(df['src'])
        self.tgt1 = df_to_values(df['tgt1'])
        self.tgt2 = df_to_values(df['tgt2'])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        tgt = self.tgt[idx]
        max_len = max(len(src), len(tgt))
        enc_input = torch.tensor(pad_array(src[1:-1], self.padding_idx, max_len), dtype=torch.long)
        dec_input = torch.tensor(pad_array(tgt[:-1], self.padding_idx, max_len), dtype=torch.long)
        dec_output = torch.tensor(pad_array(tgt[1:], self.padding_idx, max_len), dtype=torch.long)
        return enc_input, dec_input, dec_output

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size, padding_idx=self.padding_idx)
        val_loader = pad_data_loader(val_data, batch_size=batch_size, padding_idx=self.padding_idx)
        return train_loader, val_loader


"""
"""


def filter_tokens(tokens):
    for token in tokens:
        if replace_many_spaces(token.text) != ' ':
            yield token.text


def random_chars(n):
    chars = "".join([random.choice(string.ascii_letters + '_') for i in range(n)])
    return chars


def random_digits(n):
    digits = "".join([random.choice(string.digits) for i in range(n)])
    return digits


def random_any(n):
    return "".join([random.choice(string.digits + string.ascii_letters + '_') for i in range(n)])


def random_identifier():
    n = random.randint(2, 40)
    return random_chars(1) + random_any(n - 1)


def random_template(template_text):
    tmp = TemplateText(template_text)
    keys = tmp.get_keys()
    for key in keys:
        if key.startswith('id'):
            tmp.set_value(key, random_identifier())
            continue
        n = random.randint(1, 40)
        tmp.set_value(key, random_any(n))
    return tmp.to_string()


def random_linq_sql_template(template_src, template_tgt):
    template_text = template_src + '<br>' + template_tgt
    ss = random_template(template_text).split('<br>')
    return ss[0], ss[1]


def remove_file(file_path):
    os.remove(file_path) if os.path.exists(file_path) else None


train_templates = [
    'from @id1 in @id2 select @id1.@id3',
    'SELECT [@id1].[@id3] AS [@id3] FROM [dbo].[@id2] AS [@id1] WITH(NOLOCK)',

    'from @id1 in @id2 join @id4 in @id5 on @id2.@id6 equals @id1.@id7 select new { @id1.@id3, @id4.@id8 }',
    'SELECT [@id1].[@id3] AS [@id3], [@id4].[@id8] AS [@id8] FROM [dbo].[@id2] AS [@id1] WITH(NOLOCK) '
    'JOIN [dbo].[@id5] AS [@id4] WITH(NOLOCK) ON [@id2].[@id6] = [@id1].[@id7]',
]


def random_train_template():
    for idx, text in enumerate(train_templates):
        if idx % 2 == 0:
            src = text
        else:
            tgt = text
            yield src, tgt


if __name__ == '__main__':
    write_train_files()
