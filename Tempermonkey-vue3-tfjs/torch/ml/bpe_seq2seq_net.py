from torch import nn
from torch.utils.data import Dataset

from ml.lit import BaseLightning
from ml.seq2seq_net3 import Seq2SeqTransformer
from ml.bpe_tokenizer import SimpleTokenizer
from labs.preprocess_data import TranslationDataset, TranslationFileTextIterator, int_list_to_str
from utils.linq_tokenizr import linq_tokenize
from utils.tsql_tokenizr import tsql_tokenize
from dataclasses import dataclass
from typing import Callable


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
    with open('./output/linq-sample.csv', "w", encoding='UTF-8') as csv:
        csv.write('src\ttgt\n')
        for src, tgt in TranslationFileEncodeIterator('../data/linq-sample.txt'):
            csv.write(int_list_to_str(src))
            csv.write('\t')
            csv.write(int_list_to_str(tgt))
            csv.write('\n')


tk = SimpleTokenizer(None)


@dataclass(frozen=True)
class TranslateOptions:
    src_vocab_size: int
    tgt_vocab_size: int
    bos_idx: int
    eos_idx: int
    padding_idx: int
    decode_fn: Callable[[list[int]], str]
    train_dataset: Dataset


bpe_translate_options = TranslateOptions(
    src_vocab_size=tk.vocab_size,
    tgt_vocab_size=tk.vocab_size,
    bos_idx=tk.bos_idx,
    eos_idx=tk.eos_idx,
    padding_idx=tk.padding_idx,
    decode_fn=tk.decode,
    train_dataset=lambda: TranslationDataset("./output/linq-sample.csv", tk.padding_idx)
)


class BpeTranslator(BaseLightning):
    def __init__(self, options=bpe_translate_options):
        super().__init__()
        self.options = options
        self.model = Seq2SeqTransformer(options.src_vocab_size,
                                        options.tgt_vocab_size,
                                        bos_idx=options.bos_idx,
                                        eos_idx=options.eos_idx,
                                        padding_idx=options.padding_idx)
        self.criterion = nn.CrossEntropyLoss()  # reduction="none")
        self.init_dataloader(options.train_dataset(), 1)

    @staticmethod
    def prepare_train_data():
        write_train_csv_file()

    def forward(self, batch):
        enc_inputs, dec_inputs, dec_outputs = batch
        logits, enc_self_attns, dec_self_attns, dec_enc_attns = self.model(enc_inputs, dec_inputs)
        return logits, dec_outputs

    def _calculate_loss(self, data, mode="train"):
        (logits, dec_outputs), batch_idx = data
        loss = self.criterion(logits, dec_outputs.view(-1))
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        tgt_values = self.model.inference(text)
        tgt_text = self.options.decode_fn(tgt_values)
        return tgt_text



