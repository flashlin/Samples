from torch import nn

from ml.lit import BaseLightning
from ml.seq2seq_model import Seq2SeqTransformer
from ml.simple_bpe import SimpleTokenizer
from preprocess_data import TranslationDataset, TranslationFileTextIterator, int_list_to_str
from utils.linq_tokenizr import linq_tokenize
from utils.tsql_tokenizr import tsql_tokenize


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
