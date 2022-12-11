import random

import torch
from torch import nn
from torch.autograd import Variable

from common.io import info, get_file_by_lines_iter, info_error
from ml.lit import PositionalEncoding, start_train, BaseLightning, load_model
from ml.text_classifier_net import WordEmbeddingModel, TextEmbeddingModel
from ml.trans_linq2tsql import LinqToSqlVocab
from ml.translate_net import TranslateCsvDataset, convert_translate_file_to_csv

vocab = LinqToSqlVocab()


def train():
    model_type = None
    model_args = {
    }

    translate_csv_file_path = './output/linq_vlinq.csv'
    convert_translate_file_to_csv('./train_data/linq_vlinq.txt', translate_csv_file_path)
    translate_ds = TranslateCsvDataset(translate_csv_file_path, vocab)
    # start_train(model_type, model_args,
    #             translate_ds,
    #             batch_size=1,  # 32,
    #             resume_train=False,
    #             device='cuda',
    #             max_epochs=100)
    #
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
    # train()
    s1 = ['flash', 'is', 'super']
    m = TextEmbeddingModel()
    v1 = m.forward(s1)
    info(f" {v1=}")
