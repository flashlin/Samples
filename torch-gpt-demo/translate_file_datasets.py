from typing import IO
import csv
import torch
import torch.utils.data as Data
import itertools
from enum import Enum
from typing import TypeVar
import pandas as pd
from torch import nn
import torch.optim as optim
from data_utils import write_dict_to_file, load_dict_from_file
from transformer_models import Transformer
from tsql_tokenizr import tsql_tokenize
from vocabulary_utils import WordVocabulary
import ast


def read_lines_from_file_ptr(file_ptr: IO, n_lines: int):
    while True:
        lines = list(itertools.islice(file_ptr, n_lines))
        if not lines:
            break
        lines = [line.rstrip('\n') for line in lines]
        yield lines


def read_lines_from_file(file_path: str, n_lines: int = 2):
    with open(file_path, 'r', encoding='utf-8') as sr:
        line_pairs = read_lines_from_file_ptr(sr, n_lines)
        for lines in line_pairs:
            yield lines


T = TypeVar('T')


def pad_list(value_list: list[T], max_len: int, pad: T) -> list[T]:
    len_values = len(value_list)
    if len_values < max_len:
        return value_list + [pad] * (max_len - len_values)
    return value_list


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


def read_file_to_csv(file_path: str, output_csv_path: str):
    word_vob = WordVocabulary()
    sos_index = word_vob.vocab.SOS_index
    eos_index = word_vob.vocab.EOS_index
    pad_index = word_vob.vocab.PAD_index

    vocab_path = './data/vocab.txt'

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['src_input', 'tgt_input', 'tgt_output', 'encoder_input', 'decoder_input', 'decoder_output'])

        for line_pair in read_lines_from_file(file_path, n_lines=2):
            src, tgt = tuple(line_pair)
            src_tokens = tsql_tokenize(src)
            tgt_tokens = tsql_tokenize(tgt)
            src_words = [token.text for token in src_tokens]
            tgt_words = [token.text for token in tgt_tokens]

            src_values = [sos_index] + word_vob.encode_many_words(src_words) + [eos_index]
            tgt_values = [sos_index] + word_vob.encode_many_words(tgt_words) + [eos_index]

            ###
            padded_pair_list = pad_zip(src_values, tgt_values, max_len=900, pad=pad_index)
            for padded_src_values, padded_tgt1_values, padded_tgt2_values in padded_pair_list:
                padded_src = word_vob.decode_value_list(padded_src_values, isShow=True)
                padded_tgt1 = word_vob.decode_value_list(padded_tgt1_values, isShow=True)
                padded_tgt2 = word_vob.decode_value_list(padded_tgt2_values, isShow=True)
                encoder_input = remove_enum(padded_src_values)
                decoder_input = remove_enum(padded_tgt1_values)
                decoder_output = remove_enum(padded_tgt2_values)
                writer.writerow([padded_src, padded_tgt1, padded_tgt2,
                                 encoder_input, decoder_input, decoder_output])
    write_dict_to_file(word_vob.to_serializable(), vocab_path)
    return word_vob


class MyDataSet(Data.Dataset):
    def __init__(self, csv_file_path, chunk_size=1):
        super(MyDataSet, self).__init__()
        self.csv_file_path = csv_file_path
        self.chunk_size = chunk_size
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            self.headers = file.readline().strip().split(',')

    def __len__(self):
        with open(self.csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            return sum(1 for _ in csv_reader) - 1  # 減去 header 的行數

    def __getitem__(self, idx):
        data_frame = pd.read_csv(self.csv_file_path, skiprows=idx, nrows=self.chunk_size)
        dict_value = data_frame.iloc[0]
        new_dict = {}
        for index, header in enumerate(self.headers):
            new_dict[header] = dict_value[index]
        return new_dict


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


def test(model, enc_input, start_symbol):
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


def encode_text(text):
    tokens = tsql_tokenize(text)
    words = [token.text for token in tokens]
    return words


def infer(model, vocab, text):
    words = encode_text(text)
    enc_inputs = remove_enum(vocab.encode_many_words(words))
    enc_inputs = torch.LongTensor([enc_inputs]).cuda()
    start_symbol = vocab.vocab.SOS_index
    predict_dec_input = test(model, enc_inputs, start_symbol=start_symbol)
    predict, _, _, _ = model(enc_inputs, predict_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    output_values = predict.squeeze().cpu().numpy()

    output = vocab.decode_value_list(output_values, isShow=True)
    # print(f'{output_values=}')
    print(f'{output=}')


def train():
    translate_file_path = './data/tsql.txt'
    csv_file_path = './data/tsql.csv'
    word_vob = read_file_to_csv(translate_file_path, csv_file_path)
    data_set = MyDataSet(csv_file_path)
    data_loader = Data.DataLoader(data_set, batch_size=2, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(word_vob.vocab)
    model = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=2)         # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(50):
        #for enc_inputs, dec_inputs, dec_outputs in data_loader:  # enc_inputs : [batch_size, src_len]
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
    torch.save(model, './output/model.pth')
    # torch.save(model.state_dict(), 'model.pth')
    print("保存模型")


if __name__ == '__main__':
    # train()
    word_vob = WordVocabulary()
    vocab_dict = load_dict_from_file('./data/vocab.txt')
    word_vob.from_serializable(vocab_dict['token_to_idx'])
    vocab_size = len(word_vob.vocab)
    # model = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size)
    # model.load_state_dict(torch.load('./output/model.pth'))
    model = torch.load('./output/model.pth')
    infer(model, word_vob, "select id")
