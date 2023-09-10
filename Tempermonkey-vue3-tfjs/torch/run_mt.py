import argparse
import sys

from ml.lit import start_train, load_model, copy_last_ckpt
from ml.bpe_seq2seq_net import BpeTranslator
from ml.bpe_tokenizer import SimpleTokenizer
from ml.mnt_net import MntTranslator
from ml.token_type_seq2seq_net import TokenTypeTranslator
from utils.tsql_tokenizr import tsql_tokenize


def get_args():
    parser = argparse.ArgumentParser(description='simple ml')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    parser.add_argument("-n", "--name", help="model name BpeTranslator TokenTypeTranslator", dest="model_name")
    parser.add_argument("-p", "--prepare", help="optional prepare data", dest="prepare", action='store_true')
    parser.add_argument("-t", "--train", help="optional train data", dest="train", action='store_true')
    parser.add_argument("-e", "--evaluate", help="optional evaluate", dest="evaluate", action='store_true')
    parser.add_argument("-c", "--copy", help="optional copy ckpt", dest="copy", action='store_true')
    return parser.parse_args()


def prepare_data(model_type):
    print(f"start preparing data for {model_type.__name__}")
    model_type.prepare_train_data()


def train(model_type):
    print(f"start training")
    start_train(model_type, device='cuda', max_epochs=100)
    # start_train(MntTranslator, device='cuda', max_epochs=100)


def evaluate(model_type):
    print(f"test")
    model = load_model(model_type)

    def inference(text):
        print(text)
        sql = model.infer(text)
        print(sql)

    inference('from tb3 in customer select new tb3')
    inference('from c in customer select new { c.id, c.name }')
    inference('from c in customer join p in products on c.id equals p.id select new { c.id, c.name, p.name }')


def test():
    tk = SimpleTokenizer(tsql_tokenize)
    sql1 = 'SELECT name FROM customer WITH(NOLOCK)'
    print(f"{sql1=}")
    tokens = tk.encode(sql1, True)
    print(f"{tokens=}")
    sql2 = tk.decode(tokens, False)
    print(f"{sql2=}")


def str_to_class(class_name):
    return getattr(sys.modules[__name__], class_name)


if __name__ == '__main__':
    # test()
    args = get_args()
    model_type = str_to_class(args.model_name)
    if args.train:
        train(model_type)
        sys.exit(0)
    if args.prepare:
        prepare_data(model_type)
        sys.exit(0)
    if args.evaluate:
        evaluate(model_type)
        sys.exit(0)
    if args.copy:
        copy_last_ckpt(model_type.__name__)
        sys.exit(0)
