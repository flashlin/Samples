import random

import torch
from torch import nn
from torch.autograd import Variable

from common.io import info, get_file_by_lines_iter, info_error
from ml.attension_models import Transformer
from ml.gnmt.net import LiGmnTranslator
from ml.mini_gpt_net import LitMiniGpt
from ml.seq2seq_net import LiSeq2Seq
from ml.seq2seq_net2 import Seq2SeqNet
from ml.seq2seq_net3 import LitTranslator
from ml.seq2seq_net4 import LitTransformer
from ml.seq2seq_net5 import MySeq2SeqNet, get_num_segments, pad_list_by_num_segments, get_segment
from ml.seq2seq_net6 import LitTransformer2
from ml.seq_to_classification_net import SeqToOneClassificationLstm
from ml.lit import PositionalEncoding, start_train, BaseLightning, load_model
from ml.trans_linq2tsql import LinqToSqlVocab
from ml.translate_net import TranslateCsvDataset, convert_translate_file_to_csv

vocab = LinqToSqlVocab()


class MySeq(BaseLightning):
    def __init__(self):
        super().__init__()
        self.model = Transformer(num_layers=3,
                                 d_model=1024,
                                 num_heads=16,
                                 dff=2048,
                                 input_vocab_size=vocab.get_size(),
                                 target_vocab_size=vocab.get_size(),
                                 pe_input=500,
                                 pe_target=500)

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


def train():
    model_type = MySeq
    model_args = {
    }

    model_type = SeqToOneClassificationLstm
    model_args = {
        "vocab_size": vocab.get_size(),
        "padding_idx": vocab.padding_idx,
    }

    # model_type = LiSeq2Seq
    # model_args = {
    #     "src_vocab_size": vocab.get_size(),
    #     "tgt_vocab_size": vocab.get_size(),
    #     "padding_idx": vocab.padding_idx,
    # }

    # loss
    model_type = Seq2SeqNet
    model_args = {
        "vocab": vocab,
    }

    # loss 失敗又慢
    # model_type = MySeq2SeqNet
    # model_args = {
    #     "vocab": vocab,
    # }

    # loss = 0.40 不下降
    # model_type = LitTransformer
    # model_args = {
    #     'vocab': vocab,
    # }

    # loss = 240 不下降
    # model_type = LiGmnTranslator
    # model_args = {
    #     'vocab': vocab,
    # }

    # 不會弄
    # model_type = LitMiniGpt
    # model_args = {
    #     'vocab': vocab,
    # }

    model_type = LitTransformer2
    model_args = {
        'vocab': vocab,
    }

    translate_csv_file_path = './output/linq_vlinq.csv'
    convert_translate_file_to_csv('./train_data/linq_vlinq.txt', translate_csv_file_path)
    translate_ds = TranslateCsvDataset(translate_csv_file_path, vocab)
    start_train(model_type, model_args,
                translate_ds,
                batch_size=1,  # 32,
                resume_train=False,
                device='cuda',
                max_epochs=100)

    model = load_model(model_type, model_args)

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


if __name__ == '__main__':
    info(f" {vocab.get_size()}")
    train()
