from common.csv_utils import CsvWriter
from common.io import read_text_file, get_file_name, get_dir
from utils.linq_tokenizr import linq_encode, linq_tokenize
from utils.tsql_tokenizr import tsql_encode, tsql_tokenize

def write_train_csv(file: str):
    lines = read_text_file(file)
    out_file = "./output/linq-translation.csv"
    src_max_seq_length = 0
    tgt_max_seq_length = 0
    with CsvWriter(out_file) as csv:
        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                linq_values = linq_encode(line)
                src_max_seq_length = max(len(linq_values), src_max_seq_length)
                csv.write(linq_values)
            else:
                sql_values = tsql_encode(line)
                tgt_max_seq_length = max(len(sql_values), tgt_max_seq_length)
                csv.write(sql_values)
    return src_max_seq_length, tgt_max_seq_length


def get_line_str(arr):
    return ' '.join(str(number) for number in arr)

def write_train_data(file: str):
    lines = read_text_file(file)
    out_file = "./output/linq-translation.txt"
    src_max_seq_length = 0
    tgt_max_seq_length = 0
    with open(out_file, "w", encoding='UTF-8') as f:
        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                linq_values = linq_encode(line)
                src_max_seq_length = max(len(linq_values), src_max_seq_length)
                f.write(get_line_str(linq_values) + '\n')
            else:
                sql_values = tsql_encode(line)
                tgt_max_seq_length = max(len(sql_values), tgt_max_seq_length)
                f.write(get_line_str(sql_values) + '\n')
    return src_max_seq_length, tgt_max_seq_length


def flatten(l):
    return [item for sublist in l for item in sublist]

def write_tokens_data(file: str):
    lines = read_text_file(file)
    out_file = "./output/linq-translation-tokens.txt"
    def map_tokens(tokens):
        return map(lambda x: f"{{{x.type}:{x.text}}}", tokens)
    with CsvWriter(out_file) as csv:
        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                linq_tokens = linq_tokenize(line)
                tokens = map_tokens(linq_tokens)
                csv.write(tokens)
            else:
                sql_tokens = tsql_tokenize(line)
                tokens = map_tokens(sql_tokens)
                csv.write(tokens)


