import random

import torch
from torch import nn
from torch.autograd import Variable

from common.io import info, get_file_by_lines_iter, info_error
from ml.attension_models import Transformer
from ml.lit import PositionalEncoding, start_train, BaseLightning, load_model
from ml.trans_linq2tsql import LinqToSqlVocab
from ml.translate_net import TranslateCsvDataset

vocab = LinqToSqlVocab()


class MySeq(BaseLightning):
    def __init__(self):
        super().__init__()
        self.model = Transformer(num_layers=2,
                                 d_model=512,
                                 num_heads=8,
                                 dff=2048,
                                 input_vocab_size=vocab.get_size(),
                                 target_vocab_size=vocab.get_size(),
                                 pe_input=1000,
                                 pe_target=1000)

    def forward(self, batch):
        src, src_len, tgt, tgt_len = batch

        # x_hat, _ = self.model(src, tgt, None, None, None)
        x_hat, tgt_real = self.model.train_step(src, tgt)
        return x_hat, tgt_real

    def _calculate_loss(self, data, mode="train"):
        (x_hat, y), batch = data
        loss = self.model.calculate_loss(x_hat, y)
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        self.model.eval()
        values = self.model.infer(vocab, text)
        values = values.tolist()
        return vocab.decode(values)


model_type = MySeq
model_args = {
}

translate_csv_file_path = './output/linq_vlinq.csv'
translate_ds = TranslateCsvDataset(translate_csv_file_path, vocab)
model = start_train(model_type, model_args,
                    translate_ds,
                    batch_size=1,
                    device='cuda',
                    max_epochs=100)

# model = load_model(model_type, model_args)

for src, tgt in get_file_by_lines_iter('./train_data/linq_vlinq_test.txt', 2):
    src = src.rstrip()
    tgt = tgt.rstrip()
    linq_code = model.infer(src)
    tgt_expected = vocab.decode(vocab.encode(tgt)).rstrip()
    src = ' '.join(src.split(' ')).rstrip()
    print(f'src="{src}"')
    if linq_code != tgt_expected:
        info(f'"{tgt_expected}"')
        info_error(f'"{linq_code}"')
    else:
        print(f'"{linq_code}"')
    print("\n")
