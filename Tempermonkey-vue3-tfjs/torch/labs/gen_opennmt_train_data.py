import os
import re
import random
import string
from labs.preprocess_data import TranslationFileTextIterator
from utils.linq_tokenizr import linq_tokenize
from utils.template_utils import TemplateText
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
    for src, tgt in TranslationTokensIterator('../../data/linq-sample.txt'):
        mode = 'train' if random.randint(1, 10) >= 3 else 'val'
        write_train_data(mode, src, tgt)


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
    # write_train_files()

