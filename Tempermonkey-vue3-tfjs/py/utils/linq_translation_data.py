from common.csv_utils import CsvWriter
from common.io import read_text_file
from utils.linq_tokenizr import linq_encode, linq_tokenize
from utils.tsql_tokenizr import tsql_encode, tsql_tokenize


def write_train_data(file: str):
    lines = read_text_file(file)
    out_file = "./output/linq-translation.csv"
    with CsvWriter(out_file) as csv:
        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                linq_values = linq_encode(line)
                csv.write(linq_values)
            else:
                sql_values = tsql_encode(line)
                csv.write(sql_values)

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

class LinqTranslationData:
    def __init__(self):
        pass


