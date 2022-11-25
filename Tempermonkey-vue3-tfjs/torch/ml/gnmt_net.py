import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import log_softmax

from ml.bpe_tokenizer import SimpleTokenizer
from ml.lit import BaseLightning
from dataclasses import dataclass, field
from typing import Callable

from ml.seq2seq.inference.inference import Translator
from ml.seq2seq.models import GNMT
from preprocess_data import TranslationDataset

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


class GTranslator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        model = GNMT(vocab_size=vocab_size)
        translator = Translator(model,
                                tokenizer,
                                beam_size=args.beam_size,
                                max_seq_len=args.max_length_val,
                                len_norm_factor=args.len_norm_factor,
                                len_norm_const=args.len_norm_const,
                                cov_penalty_factor=args.cov_penalty_factor,
                                cuda=args.cuda)


class NmtTranslator(BaseLightning):
    def __init__(self, options=bpe_translate_options):
        super().__init__()
        self.options = options
        self.model = GNMT(vocab_size=options.tgt_vocab_size)
        self.criterion = nn.CrossEntropyLoss()
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
