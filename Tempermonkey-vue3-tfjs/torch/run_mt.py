import argparse
import sys

from ml.simple_bpe import SimpleTokenizer
from preprocess_data import TranslationFileTextIterator, int_list_to_str
from utils.linq_tokenizr import linq_tokenize
from utils.tsql_tokenizr import tsql_tokenize


def get_args():
    parser = argparse.ArgumentParser(description='simple ml')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    parser.add_argument("-p", "--prepare", help="optional prepare data", dest="prepare", action='store_true')
    parser.add_argument("-t", "--train", help="optional train data", dest="train", action='store_true')
    parser.add_argument("-e", "--test", help="optional test", dest="test", action='store_true')
    return parser.parse_args()


class TranslationFileEncodeIterator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.linq_tk = SimpleTokenizer(linq_tokenize)
        self.tsql_tk = SimpleTokenizer(tsql_tokenize)

    def __iter__(self):
        for src, tgt in TranslationFileTextIterator(self.file_path):
            src_tokens = self.linq_tk.encode(src)
            tgt_tokens = self.tsql_tk.encode(tgt)
            yield src_tokens, tgt_tokens



def prepare_data():
    print(f"start preparing data")
    with open('./output/linq-sample2.csv', "w", encoding='UTF-8') as csv:
        csv.write('src\ttgt\n')
        for src, tgt in TranslationFileEncodeIterator('../data/linq-sample.txt'):
            csv.write(int_list_to_str(src))
            csv.write('\t')
            csv.write(int_list_to_str(tgt))
            csv.write('\n')


def train():
    print(f"start training")


def test():
    print(f"test")


if __name__ == '__main__':
    args = get_args()
    if args.train:
        train()
        sys.exit(0)
    if args.prepare:
        prepare_data()
        sys.exit(0)
    if args.test:
        test()
        sys.exit(0)
