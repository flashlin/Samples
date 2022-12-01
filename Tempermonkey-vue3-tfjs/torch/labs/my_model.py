import re
import random
import string

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader

from common.io import info, remove_file, info_error
from ml.data_utils import get_data_file_path
from ml.lit import BaseLightning, start_train, copy_last_ckpt
from ml.model_utils import reduce_dim, detach_lstm_hidden_state
from utils.data_utils import df_intstr_to_values, pad_array, split_line_by_space, create_char2index_map, create_index2char_map
from utils.stream import StreamTokenIterator, read_double_quote_string_token, read_token_until, int_list_to_str, replace_many_spaces
from utils.template_utils import TemplateText


def line_to_tokens(line):
    stream_iter = StreamTokenIterator(line)
    buff = []
    prev_ch = None
    while not stream_iter.is_done():
        ch = stream_iter.peek_str(1)
        if ch == '"':
            prev_ch = read_double_quote_string_token(stream_iter).text
            buff.append(prev_ch)
            continue
        if ch == ' ':
            if prev_ch == ' ':
                prev_ch = stream_iter.next().text
                continue
            prev_ch = stream_iter.next().text
            buff.append(prev_ch)
            continue
        text = read_token_until(stream_iter, ' ').text
        prev_ch = text
        buff.append(text)
    return buff


def read_examples(example_file):
    def filter_space_tokens(a_tokens):
        for token in a_tokens:
            if token == ' ':
                continue
            yield token.rstrip()

    with open(example_file, "r", encoding='UTF-8') as f:
        for line in f:
            if line.strip() == '':
                continue
            tokens = line_to_tokens(line)
            tokens = [t for t in filter_space_tokens(tokens)]
            yield tokens


def read_examples_to_tokens3(example_file):
    for idx, tokens in enumerate(read_examples(example_file)):
        if idx % 3 == 0:
            src_tokens = tokens
            continue
        if idx % 3 == 1:
            tgt1_tokens = tokens
            continue
        tgt2_tokens = tokens
        yield src_tokens, tgt1_tokens, tgt2_tokens


def get_vocabs():
    vocab_file = get_data_file_path("linq_classification_vocab.txt")
    with open(vocab_file, "r", encoding='UTF-8') as f:
        lines = f.readlines()
        common_symbols = split_line_by_space(lines[0]) + [' ']
        src_tokens = split_line_by_space(lines[1])
        tgt_tokens = split_line_by_space(lines[2])
    return common_symbols + src_tokens, common_symbols + tgt_tokens


src_symbols, tgt_symbols = get_vocabs()
src_char2index = create_char2index_map(src_symbols)
src_index2char = create_index2char_map(src_symbols)
tgt_char2index = create_char2index_map(tgt_symbols)
tgt_index2char = create_index2char_map(tgt_symbols)


def encode_number_token(token, char2index):
    return [char2index[n] for n in token]


def encode_tokens(tokens, char2index):
    var_re = re.compile(r'(@\w.+)(\d+)')
    buff = [char2index['<bos>']]
    unk_tokens = {}
    for token in tokens:
        match = var_re.match(token)
        if match:
            name = match.group(1)
            num = match.group(2)
            buff.append(char2index[name])
            buff.append(char2index[num])
            continue
        if token not in char2index:
            unk_num = len(unk_tokens) + 1
            if token in unk_tokens:
                unk = unk_tokens[token]
            else:
                try:
                    unk_num_values = encode_number_token(str(unk_num), char2index)
                    unk = [char2index['<unk>']] + unk_num_values
                    unk_tokens[token] = unk
                except Exception as e:
                    info_error(f' {tokens=} {token=} {e}')
                    raise
            buff.extend(unk)
            continue
        buff.append(char2index[token])
    buff.append(char2index['<eos>'])
    return buff


def decode_to_text(values, index2char):
    buff = []
    for value in values:
        buff.append(index2char[value])
    return buff


def encode_src_tokens(text):
    return encode_tokens(text, src_char2index)


def encode_tgt_tokens(text):
    return encode_tokens(text, tgt_char2index)


def decode_src_to_text(text):
    return decode_to_text(text, src_index2char)


def decode_tgt_to_text(text):
    return decode_to_text(text, tgt_index2char)


def pad_row_iter(row, max_seq_len, padding_idx):
    src, tgt1, tgt2 = row
    items_lens = [len(src), len(tgt1), len(tgt2)]
    is_all_same_len = all(x == [0] for x in items_lens)
    src = pad_array(src, padding_idx, max_seq_len)
    tgt1 = pad_array(tgt1, padding_idx, max_seq_len)
    tgt2 = pad_array(tgt2, padding_idx, max_seq_len)
    if is_all_same_len:
        yield src, tgt1, tgt2
        return
    max_len = max(items_lens)
    for n in range(max_len):
        new_src = pad_array(src[n: n+max_len], padding_idx, max_seq_len)
        new_tgt1 = pad_array(tgt1[n: n+max_len], padding_idx, max_seq_len)
        new_tgt2 = pad_array(tgt2[n: n+max_len], padding_idx, max_seq_len)
        yield new_src, new_tgt1, new_tgt2


def write_train_files(max_seq_len, target_path="./output"):
    def write_train_data(a_src, a_tgt1, a_tgt2):
        f.write(int_list_to_str(a_src))
        f.write('\t')
        f.write(int_list_to_str(a_tgt1))
        f.write('\t')
        f.write(int_list_to_str(a_tgt2))
        f.write('\n')

    remove_file(f"{target_path}\\linq_sql.csv")
    example_file = get_data_file_path("linq_classification.txt")
    with open(f"{target_path}\\linq_sql.csv", "w", encoding='UTF-8') as f:
        f.write("src\ttgt1\ttgt2\n")
        for (src, tgt1, tgt2) in read_examples_to_tokens3(example_file):
            row = encode_src_tokens(src), encode_src_tokens(tgt1), encode_tgt_tokens(tgt2)
            for new_src, new_tgt1, new_tgt2 in pad_row_iter(row, max_seq_len, src_char2index['<pad>']):
                assert len(new_src) == len(new_tgt1)
                assert len(new_src) == len(new_tgt2)
                write_train_data(new_src, new_tgt1, new_tgt2)


def pad_data_loader(dataset, batch_size, padding_idx, **kwargs):
    def pad_collate(batch):
        (src_input, dec_input, dec_output) = zip(*batch)
        enc_input_pad = torch.nn.utils.rnn.pad_sequence(src_input, batch_first=True, padding_value=padding_idx)
        dec_input_pad = torch.nn.utils.rnn.pad_sequence(dec_input, batch_first=True, padding_value=padding_idx)
        dec_output_pad = torch.nn.utils.rnn.pad_sequence(dec_output, batch_first=True, padding_value=padding_idx)
        return enc_input_pad, dec_input_pad, dec_output_pad

    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=pad_collate, **kwargs)


class TranslationDataset(Dataset):
    def __init__(self, csv_file_path, padding_idx):
        self.padding_idx = padding_idx
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        self.src = df_intstr_to_values(df['src'])
        self.tgt1 = df_intstr_to_values(df['tgt1'])
        self.tgt2 = df_intstr_to_values(df['tgt2'])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        tgt1 = self.tgt1[idx]
        tgt2 = self.tgt2[idx]
        # max_len = max(len(src), len(tgt1))
        # max_len = max(len(tgt2), max_len)
        # enc_input = torch.tensor(pad_array(src[1:-1], self.padding_idx, max_len), dtype=torch.long)
        # dec_input = torch.tensor(pad_array(tgt[:-1], self.padding_idx, max_len), dtype=torch.long)
        # dec_output = torch.tensor(pad_array(tgt[1:], self.padding_idx, max_len), dtype=torch.long)
        # enc_input = torch.tensor(pad_array(src, self.padding_idx, max_len), dtype=torch.float)
        # dec_input = torch.tensor(pad_array(tgt1, self.padding_idx, max_len), dtype=torch.float)
        # dec_output = torch.tensor(pad_array(tgt2, self.padding_idx, max_len), dtype=torch.float)
        enc_input = torch.tensor(src, dtype=torch.long)
        dec_input = torch.tensor(tgt1, dtype=torch.long)
        dec_output = torch.tensor(tgt2, dtype=torch.long)
        return enc_input, dec_input, dec_output

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size, padding_idx=self.padding_idx)
        val_loader = pad_data_loader(val_data, batch_size=batch_size, padding_idx=self.padding_idx)
        return train_loader, val_loader


class LSTMTagger(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, hidden_feature_dim, hidden_layer_num, classes_num, batch_size):
        """
        :param embedding_dim: 輸入訊號的維度
        :param hidden_feature_dim: 設定越多代表能記住的特徵越多
        :param hidden_layer_num: 記住的有用的 hidden_layer_num
        :param classes_num:
        :param batch_size:
        """
        super().__init__()
        self.input_feature_dim = embedding_dim
        self.hidden_feature_dim = hidden_feature_dim
        self.hidden_layer_num = hidden_layer_num
        self.batch_size = batch_size

        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)

        # out: (batch_size=1, input_len, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_feature_dim,
                            num_layers=hidden_layer_num,
                            batch_first=True)
        self.hidden = None
        # self.linear = nn.Linear(hidden_feature_dim, classes_num)
        self.linear = nn.Sequential(
            nn.Linear(hidden_feature_dim, classes_num),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCELoss()
        # self.loss_fn = nn.NLLLoss()

    def init_hidden(self, device):
        """
        由於LSTM 的輸入必須有前一個LSTM 計算出來的hidden state 和cell state
        因此在頭一個CELL 的時候，必須自己弄一個隨機生成的初始狀態
        :return:
        """
        if self.hidden is None:
            h0 = torch.randn(self.hidden_layer_num, self.batch_size, self.hidden_feature_dim).to(device)
            c0 = torch.randn(self.hidden_layer_num, self.batch_size, self.hidden_feature_dim).to(device)
            self.hidden = h0, c0
        # h0 = torch.randn(self.batch_size, self.hidden_layer_num, self.hidden_feature_dim)
        # c0 = torch.randn(self.batch_size, self.hidden_layer_num, self.hidden_feature_dim)
        # h0 = torch.randn(self.hidden_layer_num, self.hidden_feature_dim).to(device)
        # c0 = torch.randn(self.hidden_layer_num, self.hidden_feature_dim).to(device)
        # h0 = torch.zeros(self.hidden_layer_num, self.hidden_feature_dim).to(device)
        # c0 = torch.zeros(self.hidden_layer_num, self.hidden_feature_dim).to(device)
        return self.hidden

    def forward(self, x):
        """
        lstm 的輸出會有output 和最後一個CELL 的 hidden state, cell state
        在pytorch 裏output 的值其實就是每個cell 的hidden state集合起來的陣列
        :param x:
        :return:
        """
        x = self.embedding(x)
        self.init_hidden(x.device)
        output, self.hidden = self.lstm(x, self.hidden)
        # output = self.linear(output[-1])
        output = reduce_dim(output)
        output = self.linear(output)
        # _, predictive_value = torch.max(output, 1)  # 從output中取最大的出來作為預測值
        # predictive_value = F.log_softmax(output, dim=1)  # 從output中取最大的出來作為預測值
        info(f" { reduce_dim(output)[-1]=}")
        self.hidden = detach_lstm_hidden_state(self.hidden)
        return output

    def calculate_loss(self, x, y):
        y = reduce_dim(y)
        return self.loss_fn(x, y)


class MyModel(BaseLightning):
    def __init__(self):
        super().__init__()
        batch_size = 1
        self.model = LSTMTagger(src_vocab_size=len(src_symbols),
                                embedding_dim=3,
                                hidden_feature_dim=len(src_symbols),
                                hidden_layer_num=3,
                                classes_num=len(src_symbols),
                                batch_size=batch_size)
        self.init_dataloader(TranslationDataset("../output/linq_sql.csv", src_char2index['<pad>']), batch_size)

    def forward(self, batch):
        enc_inputs, dec_inputs, dec_outputs = batch
        logits = self.model(enc_inputs)
        return logits, dec_inputs

    def _calculate_loss(self, data, mode="train"):
        (logits, dec_inputs), batch = data
        loss = self.model.calculate_loss(logits, dec_inputs)
        self.log("%s_loss" % mode, loss)
        return loss

    # def infer(self, text):
    #     sql_values = self.model.inference(text)
    #     sql = tsql_decode(sql_values)
    #     return sql


"""
----------------
"""


def filter_tokens(tokens):
    for token in tokens:
        if replace_many_spaces(token.text) != ' ':
            yield token.text


def random_chars(n):
    chars = "".join([random.choice(string.ascii_letters + '_') for i in range(n)])
    return chars


def random_digits(n):
    digits = "".join([random.choice(string.digits) for i in range(n)])
    return digits


def random_any(n):
    return "".join([random.choice(string.digits + string.ascii_letters + '_') for i in range(n)])


def random_identifier():
    n = random.randint(2, 40)
    return random_chars(1) + random_any(n - 1)


def random_template(template_text):
    tmp = TemplateText(template_text)
    keys = tmp.get_keys()
    for key in keys:
        if key.startswith('id'):
            tmp.set_value(key, random_identifier())
            continue
        n = random.randint(1, 40)
        tmp.set_value(key, random_any(n))
    return tmp.to_string()


def random_linq_sql_template(template_src, template_tgt):
    template_text = template_src + '<br>' + template_tgt
    ss = random_template(template_text).split('<br>')
    return ss[0], ss[1]


train_templates = [
    'from @id1 in @id2 select @id1.@id3',
    'SELECT [@id1].[@id3] AS [@id3] FROM [dbo].[@id2] AS [@id1] WITH(NOLOCK)',

    'from @id1 in @id2 join @id4 in @id5 on @id2.@id6 equals @id1.@id7 select new { @id1.@id3, @id4.@id8 }',
    'SELECT [@id1].[@id3] AS [@id3], [@id4].[@id8] AS [@id8] FROM [dbo].[@id2] AS [@id1] WITH(NOLOCK) '
    'JOIN [dbo].[@id5] AS [@id4] WITH(NOLOCK) ON [@id2].[@id6] = [@id1].[@id7]',
]


def random_train_template():
    for idx, text in enumerate(train_templates):
        if idx % 2 == 0:
            src = text
        else:
            tgt = text
            yield src, tgt


if __name__ == '__main__':
    MAX_SEQ_LEN = 100
    print("prepare train data...")
    write_train_files(max_seq_len=MAX_SEQ_LEN)
    copy_last_ckpt(model_name=MyModel.__name__)
    print("start training...")
    start_train(MyModel, device='cuda', max_epochs=1)
