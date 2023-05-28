import csv
import os

import torch
import torch.utils.data as Data
from enum import Enum
from typing import TypeVar
from torch import nn
import torch.optim as optim
from data_utils import write_dict_to_file, load_dict_from_file, pad_list, T
from stream_utils import read_lines_from_file, CsvDataSet
from transformer_models import Transformer
from tsql_tokenizr import tsql_tokenize, SqlVocabulary
from vocabulary_utils import WordVocabulary
import ast


def pad_zip(src_words: list[T], tgt_words: list[T],
            max_len: int, pad: T) -> list[[T, T, T]]:
    """
    :param src_words: ['a', 'b', 'c']
    :param tgt_words: ['a', 'b', 'c']
    :param max_len: 2
    :param pad: '<PAD>'
    :return: [['a','b'],['b','c']]
    """
    len_range = max(len(src_words), len(tgt_words))
    if len_range < max_len:
        len_range = 1
    if len_range > max_len:
        len_range = len_range - max_len + 1
    result = []
    for i in range(len_range):
        sub_src_words = src_words[i:i + max_len]
        sub_src = pad_list(sub_src_words, max_len, pad)
        sub_tgt1 = pad_list(tgt_words[i:i + max_len], max_len, pad)
        sub_tgt2 = pad_list(tgt_words[i + 1:i + 1 + max_len], max_len, pad)
        result.append([sub_src, sub_tgt1, sub_tgt2])
    return result


def remove_enum(value_list):
    return [item.value if isinstance(item, Enum) else item for item in value_list]


def collate_fn(batch):
    encoder_input_batch = []
    decoder_input_batch = []
    decoder_output_batch = []
    for item in batch:
        encoder_input = ast.literal_eval(item["encoder_input"])
        decoder_input = ast.literal_eval(item['decoder_input'])
        decoder_output = ast.literal_eval(item['decoder_output'])
        encoder_input_batch.append(encoder_input)
        decoder_input_batch.append(decoder_input)
        decoder_output_batch.append(decoder_output)
    return torch.LongTensor(encoder_input_batch), \
        torch.LongTensor(decoder_input_batch), \
        torch.LongTensor(decoder_output_batch)


def eval_model(model, enc_input, start_symbol):
    tgt_len = 900
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def encode_sql(text):
    tokens = tsql_tokenize(text)
    words = [token.text for token in tokens]
    return words


def infer(model, vocab, text):
    words = encode_sql(text)
    enc_input = remove_enum(vocab.encode_many_words(words))
    enc_input = torch.LongTensor([enc_input]).cuda()
    start_symbol = vocab.vocab.SOS_index
    predict_dec_input = eval_model(model, enc_input, start_symbol=start_symbol)
    predict, _, _, _ = model(enc_input, predict_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    output_values = predict.squeeze().cpu().numpy()

    output = vocab.decode_value_list(output_values, isShow=True)
    # print(f'{output_values=}')
    print(f'{output=}')


class SqlTransformer:
    def __init__(self):
        self.vocab = SqlVocabulary()
        self.model_pt_file_path = './output/model.pth'
        self.vocab_file_path = './output/vocab.txt'
        self.model = None

    def load_model(self):
        self.vocab.load(self.vocab_file_path)
        vocab_size = len(self.vocab)
        self.model = model = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).cuda()
        if os.path.exists(self.model_pt_file_path):
            model.load_state_dict(torch.load(self.model_pt_file_path))
        return model

    def convert_translate_file_to_csv_file(self, translate_file_path: str, csv_file_path: str):
        if os.path.exists(csv_file_path):
            return
        word_vocab = self.vocab
        sos_index = self.vocab.SOS_index
        eos_index = self.vocab.EOS_index
        pad_index = self.vocab.PAD_index
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['src_input', 'tgt_input', 'tgt_output', 'encoder_input', 'decoder_input', 'decoder_output'])
            for line_pair in read_lines_from_file(translate_file_path, n_lines=2):
                src, tgt = tuple(line_pair)
                src_tokens = tsql_tokenize(src)
                tgt_tokens = tsql_tokenize(tgt)
                src_words = [token.text for token in src_tokens]
                tgt_words = [token.text for token in tgt_tokens]
                src_values = [sos_index] + word_vocab.encode_many_words(src_words) + [eos_index]
                tgt_values = [sos_index] + word_vocab.encode_many_words(tgt_words) + [eos_index]
                padded_pair_list = pad_zip(src_values, tgt_values, max_len=900, pad=pad_index)
                for padded_src_values, padded_tgt1_values, padded_tgt2_values in padded_pair_list:
                    padded_src = word_vocab.decode_value_list(padded_src_values, is_show=True)
                    padded_tgt1 = word_vocab.decode_value_list(padded_tgt1_values, is_show=True)
                    padded_tgt2 = word_vocab.decode_value_list(padded_tgt2_values, is_show=True)
                    encoder_input = remove_enum(padded_src_values)
                    decoder_input = remove_enum(padded_tgt1_values)
                    decoder_output = remove_enum(padded_tgt2_values)
                    writer.writerow([padded_src, padded_tgt1, padded_tgt2,
                                     encoder_input, decoder_input, decoder_output])
        word_vocab.save(self.vocab_file_path)

    def train(self, csv_file_path, max_epoch=50):
        data_set = CsvDataSet(csv_file_path)
        data_loader = Data.DataLoader(data_set, batch_size=2, shuffle=True, collate_fn=collate_fn)
        model = self.load_model()
        criterion = nn.CrossEntropyLoss(ignore_index=2)  # 忽略 占位符 索引为0.
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
        for epoch in range(max_epoch):
            for batch in data_loader:  # enc_inputs : [batch_size, src_len]
                enc_inputs, dec_inputs, dec_outputs = batch
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                # outputs: [batch_size * tgt_len, tgt_vocab_size]
                loss = criterion(outputs, dec_outputs.view(-1))
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                torch.save(model.state_dict(), self.model_pt_file_path)


def train():
    translate_file_path = './data/tsql.txt'
    csv_file_path = './data/tsql.csv'
    m = SqlTransformer()
    m.convert_translate_file_to_csv_file(translate_file_path, csv_file_path)
    m.train(csv_file_path)


def test():
    word_vob = WordVocabulary()
    vocab_dict = load_dict_from_file('./data/vocab.txt')
    word_vob.from_serializable(vocab_dict['token_to_idx'])
    vocab_size = len(word_vob.vocab)
    # model = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size)
    # model.load_state_dict(torch.load('./output/model.pth'))
    model = torch.load('./output/model.pth')
    infer(model, word_vob, "select id")


if __name__ == '__main__':
    train()
    # test()
