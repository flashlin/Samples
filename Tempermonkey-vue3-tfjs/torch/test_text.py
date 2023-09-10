import random

import torch
from torch import nn
from torch.autograd import Variable

from common.io import info, get_file_by_lines_iter, info_error
from ml.bpe_utils import BPE, generate_bpe_vocabulary_and_merges, TextEncoding
from ml.lit import load_model
from ml.text_classifier_net import WordEmbeddingModel, TextEmbeddingModel, TextClassifier, ProgramLangVocab
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


def test_train1(text, n_class):
    info(f" {text=}")
    classifier = TextClassifier()
    logits = classifier.forward(text)
    info(f" {logits=}")
    tgt = torch.tensor(n_class)
    loss = classifier.calculate_loss((logits, tgt))
    info(f" {loss=}")
    prediction = classifier.infer(text)
    info(f" infer {prediction=}")


def test_bpe():
    bpe = BpeVocab()

    with open('train_data/linq.txt', 'r', encoding='utf-8') as f:
        raw_list_of_text = f.readlines()
    bpe.create_train_csv_file(raw_list_of_text)
    bpe.train()
    bpe.load()
    # Encode a sentence using BPE
    encoded_sentence = bpe.encode_tokens(['from', 'flash', 'from_flash', 'From'])
    # Decode the encoded sentence using BPE
    # decoded_sentence = bpe.decode(encoded_sentence)
    info(f" {encoded_sentence=}")
    s1 = bpe.decode_values(encoded_sentence)
    info(f" {s1=}")


def test():
    test_train1('from tb1 in customer select tb1', 1)
    test_train1('from tb1 in customer join tb2 in home select new { tb1.name, tb2.addr}', 2)


def test_bpe():
    words = ['low', 'lower', 'sh']
    p = BPE()
    p.build(words)
    print(f" {p.char_frequencies=}")
    print(f" {p.vocab_tokenization=}")
    print(f" {p.token2index=}")
    def encode(word):
        encode_values = p.encode(word)
        info(f" {word=} {encode_values=}")
        s1 = p.decode(encode_values)
        info(f" {s1=}")
    encode('flash')
    encode('lower')

if __name__ == '__main__':
    info(f" {vocab.get_size()}")
    # train()
    # test()
    # test_bpe()
    #words.append('1234567890')
    # words.append('abcdefghijklmnopqrstuvwxyz')
    # words.append('~!@#$%^&*()_+`_+{}|{}|:";\'<>?,./')
    # v.merge()
    # info(f" {v.word_freq=}")
    # info(f" {v.pairs=}")
    # v1 = v.encode('flash')
    # info(f" {v1=}")

    vocab = ProgramLangVocab()
    model = TextEncoding(vocab)
    model.read_text_file('train_data/linq.txt')
    model.bpe.add_word('f')
    model.bpe.add_word('la')
    model.bpe.add_word('sh')
    # model.read_cvc_file()
    model.build()
    s1 = "from tb1 in customer"
    v1 = model.encode(s1)
    print(f" {s1=}")
    print(f" {v1=}")
    s2 = model.decode(v1)
    print(f" {s2=}")
    s1 = 'flash'
    v1 = model.encode(s1)
    s2 = model.decode(v1)
    print(f" {s1=} {s2=}")

