import random

import torch
from torch import nn
from torch.utils.data import Dataset, random_split

from common.io import info
from ml.lit import PositionalEncoding, BaseLightning, load_model, copy_last_ckpt, start_train
from ml.mnt_net import pad_data_loader
from ml.model_utils import reduce_dim
from ml.translate_net import Seq2SeqTransformer
from my_model import encode_tokens, decode_to_text
from utils.stream import StreamTokenIterator, read_double_quote_string_token, read_token_until, read_identifier_token, \
    EmptyToken, \
    read_symbol_token, read_spaces_token, Token, reduce_token_list
from utils.data_utils import sort_desc, create_char2index_map, create_index2char_map

"""
src = 'from tb1     in customer select tb1     . name'
tgt = 'from @tb_as1 in @tb1     select @tb_as1 . @fd1'

src = 'from tb1     in'
pre = '<bos>'
tgt = 'from @tb_as1 in'

src = 'tb1     in customer'
pre = 'from'
tgt = '@tb_as1 in @tb1'

src = 'in customer select'
pre = 'from in'
tgt = 'in @tb1     select'

src = 'customer select tb1'
pre = 'from in customer'
tgt = '@tb1     select @tb_as1'

src = 'select tb1     .'
pre = 'in customer select'
tgt = 'select @tb_as1 .'

src = 'tb1     . name'
pre = 'customer select tb1'
tgt = '@tb_as1 . @fd1'

src = '. name <eos>'
pre = 'select tb1 .'
tgt = '. @fd1 <eos>'
"""

src = 'from tb1 in customer select tb1.name'
pre = ''
tgt = 'from @tb_as1 in @tb1 select @tb_as1.@fd1'


class LinqToSqlVocab:
    def __init__(self):
        self.symbols = symbols = '. [ ] { } += + - * / , =='.split(' ')
        common_symbols = '1 2 3 4 5 6 7 8 9 0 <unk> <bos> <eos> <pad>'.split(' ') + [' ']
        linq_spec = 'from in select new join on equals contains'.split(' ')
        linq_symbols = sort_desc(common_symbols + symbols + linq_spec)
        tsql_spec = '@tb_as @tb @fd_as @fd @str @number'.split(' ')
        tsql_symbols = sort_desc(common_symbols + symbols + tsql_spec)
        self.shared_symbols = shared_symbols = sort_desc(common_symbols + symbols + linq_symbols + tsql_symbols)
        self.char2index = create_char2index_map(shared_symbols)
        self.index2char = create_index2char_map(shared_symbols)

    def get_size(self):
        return len(self.shared_symbols)

    def encode_tokens(self, tokens: [str]) -> [int]:
        return encode_tokens(tokens, self.char2index)

    def decode_values(self, values: [int]) -> [str]:
        return decode_to_text(values, self.index2char)

    @staticmethod
    def read_variable_token(stream_iter: StreamTokenIterator) -> Token:
        if stream_iter.peek_str(1) != '@':
            return EmptyToken
        at_token = stream_iter.next()
        token = read_identifier_token(stream_iter)
        return reduce_token_list('variable', [at_token, token])

    def encode_to_tokens(self, line) -> [str]:
        stream_iter = StreamTokenIterator(line)
        buff = []
        while not stream_iter.is_done():
            token = LinqToSqlVocab.read_variable_token(stream_iter)
            if token != EmptyToken:
                buff.append(token.text)
                continue
            token = read_double_quote_string_token(stream_iter)
            if token != EmptyToken:
                buff.append(token.text)
                continue
            token = read_spaces_token(stream_iter)
            if token != EmptyToken:
                buff.append(' ')
                continue
            token = read_symbol_token(stream_iter, self.symbols)
            if token != EmptyToken:
                buff.append(token.text)
                continue
            token = read_identifier_token(stream_iter)
            if token != EmptyToken:
                buff.append(token.text)
                continue

            text = read_token_until(stream_iter, ' ').text
            buff.append(text)
        return buff

    def encode(self, text: str) -> [int]:
        tokens = self.encode_to_tokens(text)
        return self.encode_tokens(tokens)

    def get_value(self, char: str) -> int:
        return self.char2index[char]


vocab = LinqToSqlVocab()
src_tokens = vocab.encode_to_tokens(src)
print(f"{src_tokens=}")
src_values = vocab.encode_tokens(src_tokens)
print(f"{src_values=}")
src_text = vocab.decode_values(src_values)
print(f"{src_text=}")
tgt_tokens = vocab.encode_to_tokens(tgt)
print(f"{tgt_tokens=}")
tgt_values = vocab.encode_tokens(tgt_tokens)

SEQ_LEN = 3
SRC_WORD_DIM = 3
TGT_WORD_DIM = 3
POS_DIM = 3
MAX_SENTENCE_LEN = 1000
HIDDEN_SIZE = 7
NUM_LAYERS = 3


class Encoder(nn.Module):
    def __init__(self,
                 pos_dim, max_sentence_len,
                 src_vocab_size, src_word_dim,
                 src_padding_idx,
                 hidden_size=3,
                 dropout=0.1,
                 num_layers=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.src_embedding = nn.Embedding(num_embeddings=src_vocab_size,
                                          embedding_dim=src_word_dim,
                                          padding_idx=src_padding_idx)
        self.pos_emb = PositionalEncoding(d_model=pos_dim, max_len=max_sentence_len)
        self.lstm = nn.LSTM(input_size=src_word_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

    def forward(self, x):
        x = self.src_embedding(x)
        x = self.pos_emb(x)
        output, hidden = self.lstm(x)  # h_n = (num_layers * num_directions, batch_size, hidden_size)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self,
                 seq_len, pos_dim, max_sentence_len,
                 tgt_vocab_size, tgt_word_dim, tgt_padding_idx,
                 hidden_size=3, num_layers=7, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tgt_embedding = nn.Embedding(num_embeddings=tgt_vocab_size,
                                          embedding_dim=tgt_word_dim,
                                          padding_idx=tgt_padding_idx)
        self.pos_emb = PositionalEncoding(d_model=pos_dim, max_len=max_sentence_len)
        self.lstm = nn.LSTM(input_size=tgt_word_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.classify = nn.Linear(hidden_size, tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, tgt, hidden):
        """
        :param trg: [batch, seq_len] 實際訓練應該為 seq_len = 1
        :param hidden: [n_layers, batch, hidden_dim]
        :return:
        """
        # tgt = tgt.unsqueeze(1)
        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_emb(tgt)
        print(f" decoder {tgt.shape=} {hidden[0].shape=}")
        output, hidden = self.lstm(tgt, hidden)
        predictive = self.classify(output)
        return predictive, hidden


# （L - 長度，N - 批量大小，F - 特徵大小）

class Seq2Seq(nn.Module):
    def __init__(self,
                 seq_len, src_pos_dim, max_sentence_len,
                 src_vocab_size, src_word_dim, src_padding_idx,
                 tgt_vocab_size, tgt_word_dim, tgt_padding_idx,
                 hidden_size=3,
                 num_layers=7,
                 dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder = Encoder(pos_dim=src_pos_dim,
                               max_sentence_len=max_sentence_len,
                               src_vocab_size=src_vocab_size,
                               src_word_dim=src_word_dim,
                               src_padding_idx=src_padding_idx,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout)
        self.decoder = Decoder(seq_len=seq_len,
                               pos_dim=src_pos_dim,
                               max_sentence_len=max_sentence_len,
                               tgt_vocab_size=tgt_vocab_size,
                               tgt_word_dim=tgt_word_dim,
                               tgt_padding_idx=tgt_padding_idx,
                               num_layers=num_layers,
                               hidden_size=hidden_size,
                               dropout=dropout)
        self.out2tag = nn.Linear(hidden_size, tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_padding_idx)

    def forward(self, src, tgt, teach_rate=0.5):
        batch_size = tgt.size(0)
        tgt_seq_len = tgt.size(1)

        outputs_save = torch.zeros(batch_size, tgt_seq_len, self.tgt_vocab_size)
        _, hidden = self.encoder(src)
        info(f" cnoder {hidden[0].shape=}")

        tgt_i = tgt[:, 0]
        for i in range(0, tgt_seq_len):
            print(f" {tgt_i.shape=} {hidden[0].shape=}")
            output, hidden = self.decoder(tgt_i, hidden)
            outputs_save[:, i, :] = output
            top = output.argmax(dim=0).argmax(dim=0)  # 拿到預測結果
            top = max(top)
            probability = random.random()
            if probability > teach_rate:
                info(f" teach {tgt[:, i].shape=}")
                tgt_i = tgt[:, i]
            else:
                print(f" top {top.shape=} {top=}")
                tgt_i = top
            # tgt_i = tgt[:, i] if probability > teach_rate else top
        return outputs_save

    def calculate_loss(self, x_hat, y):
        # x_hat = x_hat.contiguous().view(-1, x_hat.size(-1))
        # y = y.contiguous().view(-1)
        x_hat = x_hat[:, 1:, :].reshape(-1, self.tgt_vocab_size)
        y = y[:, 1:].reshape(-1)
        return self.loss_fn(x_hat, y)


inp1 = src_values  # [0:3]
inp2 = tgt_values  # [0:3]

inp1_values = torch.tensor([inp1], dtype=torch.long)
inp2_values = torch.tensor([inp2], dtype=torch.long)

# model = Seq2Seq(seq_len=SEQ_LEN,
#                 src_pos_dim=POS_DIM,
#                 max_sentence_len=MAX_SENTENCE_LEN,
#                 src_vocab_size=SRC_VOCAB_SIZE,
#                 src_word_dim=SRC_WORD_DIM,
#                 src_padding_idx=SRC_PADDING_IDX,
#                 tgt_vocab_size=TGT_VOCAB_SIZE,
#                 tgt_word_dim=TGT_WORD_DIM,
#                 tgt_padding_idx=TGT_PADDING_IDX,
#                 )

# predictive, y = model(inp1_values, inp2_values)
# print(f"{predictive=}")
#
# loss = model.calculate_loss(predictive, y)
# print(f" {loss=}")


class MemDataset(Dataset):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx
        self.data1 = [src_values]
        self.data2 = [tgt_values]

    def __len__(self):
        # return len(self.data1)
        return 100

    def __getitem__(self, idx):
        src = self.data1[0]
        tgt = self.data2[0]
        src = torch.tensor(src, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        return src, len(src), tgt, len(tgt)

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size, padding_idx=self.padding_idx)
        val_loader = pad_data_loader(val_data, batch_size=batch_size, padding_idx=self.padding_idx)
        return train_loader, val_loader


class MyTrans(BaseLightning):
    def __init__(self):
        super().__init__()
        self.model = Seq2SeqTransformer(vocab.get_size(), vocab.get_value('<pad>'))
        batch_size = 1
        self.init_dataloader(MemDataset(vocab.get_value('<pad>')), batch_size)

    def forward(self, batch):
        src, src_len, tgt, tgt_len = batch

        tgt_y = tgt[:, 1:]
        tgt = tgt[:, :-1]

        x_hat = self.model(src, tgt)
        return x_hat, tgt_y

    def _calculate_loss(self, data, mode="train"):
        (x_hat, y_hat), batch = data
        loss = self.model.calculate_loss(x_hat, y_hat)
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text, vocab):
        src_values = vocab.encode(text)
        self.model.eval()
        device = next(self.parameters()).device
        src = torch.tensor([src_values], dtype=torch.long).to(device)
        bos = vocab.get_value('<bos>')
        tgt = torch.tensor([[bos]], dtype=torch.long).to(device)
        for i in range(len(src_values)):
            outputs = self.model.transform(src, tgt)
            # 預測結果，因為只需要看最後一個詞，所以取`out[:, -1]`
            last_word = outputs[:, -1]
            predict = self.model.predictor(last_word)
            # 找出最大值的 index
            y = torch.argmax(predict, dim=1)
            # 和之前的預測結果拼接到一起
            tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
            if y == vocab.get_value('<eos>'):
                break

        result = vocab.decode_values(reduce_dim(tgt).tolist())
        return result


copy_last_ckpt(MyTrans)
# model = start_train(MyTrans, device='cuda', max_epochs=100)
model = load_model(MyTrans)
text = model.infer('from tb2 in p select tb2.name', vocab)
print(f"{text=}")
