import os
import random
import re
import random
import string

from common.io import info
from preprocess_data import TranslationFileTextIterator
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


def read_examples_to_tokens_tuple(example_file):
    for idx, tokens in enumerate(read_examples(example_file)):
        if idx % 3 == 0:
            src_tokens = tokens
            continue
        if idx % 3 == 1:
            decode1_tokens = tokens
            continue
        decode2_tokens = tokens
        yield src_tokens, decode1_tokens, decode2_tokens


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
    def write_train_data(mode, src, de1_values, de2_values):
        with open(f"{target_path}\\linq_sql-{mode}.txt", "a+", encoding='UTF-8') as f:
            f.write(int_list_to_str(src))
            f.write('\n')
            f.write(int_list_to_str(de1_values))
            f.write('\n')
            f.write(int_list_to_str(de2_values))

    remove_file(f"{target_path}\\src-train.txt")
    remove_file(f"{target_path}\\src-val.txt")
    file = get_data_file_path("linq_classification.txt")
    for (src, de1, de2) in read_examples_to_tokens_tuple(file):
        src_values = encode_src(src)
        de1_values = encode_src(de1)
        de2_values = encode_tgt(de2)
        mode = 'train' if random.randint(1, 10) >= 3 else 'val'
        write_train_data(mode, src_values, de1_values, de2_values)


"""
"""


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
