import argparse
import sys

from ml.lit import start_train, load_model, copy_last_ckpt
from ml.bpe_seq2seq_net import BpeTranslator, write_train_csv_file
from ml.simple_bpe import SimpleTokenizer
from utils.tsql_tokenizr import tsql_tokenize


def get_args():
    parser = argparse.ArgumentParser(description='simple ml')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    parser.add_argument("-p", "--prepare", help="optional prepare data", dest="prepare", action='store_true')
    parser.add_argument("-t", "--train", help="optional train data", dest="train", action='store_true')
    parser.add_argument("-e", "--test", help="optional test", dest="test", action='store_true')
    parser.add_argument("-c", "--copy", help="optional copy ckpt", dest="copy", action='store_true')
    return parser.parse_args()


def prepare_data():
    print(f"start preparing data")
    write_train_csv_file()


def train():
    print(f"start training")
    start_train(BpeTranslator, device='cuda', max_epochs=100)


def evaluate():
    print(f"test")
    model = load_model(BpeTranslator)

    def inference(text):
        print(text)
        sql = model.infer(text)
        print(sql)

    inference('from tb3 in customer select new tb3')
    inference('from c in customer select new { c.id, c.name }')


def test():
    tk = SimpleTokenizer(tsql_tokenize)
    sql1 = 'SELECT name FROM customer WITH(NOLOCK)'
    print(f"{sql1=}")
    tokens = tk.encode(sql1, True)
    print(f"{tokens=}")
    sql2 = tk.decode(tokens, False)
    print(f"{sql2=}")


if __name__ == '__main__':
    # test()
    args = get_args()
    if args.train:
        train()
        sys.exit(0)
    if args.prepare:
        prepare_data()
        sys.exit(0)
    if args.test:
        evaluate()
        sys.exit(0)
    if args.copy:
        copy_last_ckpt()
        sys.exit(0)
