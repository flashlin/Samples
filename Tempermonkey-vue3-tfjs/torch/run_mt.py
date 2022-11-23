import argparse
import sys

from torch import nn

from common.io import get_directory_list_by_pattern, get_file_list_by_pattern
from lit import BaseLightning, start_train, load_model
from ml.seq2seq_model import Seq2SeqTransformer
from ml.simple_bpe import SimpleTokenizer
from preprocess_data import TranslationFileTextIterator, int_list_to_str, TranslationDataset
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
    parser.add_argument("-c", "--copy", help="optional copy ckpt", dest="copy", action='store_true')
    return parser.parse_args()


class TranslationFileEncodeIterator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.linq_tk = SimpleTokenizer(linq_tokenize)
        self.tsql_tk = SimpleTokenizer(tsql_tokenize)

    def __iter__(self):
        for src, tgt in TranslationFileTextIterator(self.file_path):
            src_tokens = self.linq_tk.encode(src, add_start_end=True)
            tgt_tokens = self.tsql_tk.encode(tgt, add_start_end=True)
            yield src_tokens, tgt_tokens


def write_train_csv_file():
    with open('./output/linq-sample2.csv', "w", encoding='UTF-8') as csv:
        csv.write('src\ttgt\n')
        for src, tgt in TranslationFileEncodeIterator('../data/linq-sample.txt'):
            csv.write(int_list_to_str(src))
            csv.write('\t')
            csv.write(int_list_to_str(tgt))
            csv.write('\n')


class BpeTranslator(BaseLightning):
    def __init__(self):
        super().__init__()
        self.tk = tk = SimpleTokenizer(None)
        self.model = Seq2SeqTransformer(tk.vocab_size, tk.vocab_size,
                                        bos_idx=tk.bos_idx,
                                        eos_idx=tk.eos_idx,
                                        padding_idx=tk.padding_idx)
        self.criterion = nn.CrossEntropyLoss()  # reduction="none")
        self.init_dataloader(TranslationDataset("./output/linq-sample2.csv", tk.padding_idx), 1)

    def forward(self, batch):
        enc_inputs, dec_inputs, dec_outputs = batch
        logits, enc_self_attns, dec_self_attns, dec_enc_attns = self.model(enc_inputs, dec_inputs)
        return logits, dec_outputs

    def _calculate_loss(self, data, mode="train"):
        (logits, dec_outputs), batch = data
        loss = self.criterion(logits, dec_outputs.view(-1))
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        sql_values = self.model.inference(text)
        sql = self.tk.decode(sql_values)
        return sql


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


class LightningLogsIterator:
    def __init__(self, lightning_logs_path):
        self.lightning_logs_path = lightning_logs_path + "/lightning_logs"

    def __iter__(self):
        for folder in get_directory_list_by_pattern(self.lightning_logs_path, r'version_\d+'):
            for ckpt in get_file_list_by_pattern(folder + '/checkpoints', r'.+\.ckpt'):
                yield ckpt


def copy_last_cpk():
    for ckpt in LightningLogsIterator('./output/BpeTranslator'):
        print(f"{ckpt=}")


def test():
    tk = SimpleTokenizer(tsql_tokenize)
    sql1 = 'SELECT name FROM customer WITH(NOLOCK)'
    print(f"{sql1=}")
    tokens = tk.encode(sql1, True)
    print(f"{tokens=}")
    sql2 = tk.decode(tokens, False)
    print(f"{sql2=}")


if __name__ == '__main__':
    #test()
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
        copy_last_cpk()
        sys.exit(0)

