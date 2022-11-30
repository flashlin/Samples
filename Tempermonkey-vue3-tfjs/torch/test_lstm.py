import random

import torch
from torch import nn

from common.io import info
from ml.lit import PositionalEncoding, load_model, copy_last_ckpt, start_train
from ml.trans_linq2tsql import LinqToSqlVocab
from ml.translate_net import LiTranslator, ListDataset

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
tgt = 'from @tb_as1 in @tb1 select @tb_as1.@fd1'

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


translate_examples = [
    (
        'from tb1     in customer select tb1.name',
        'from @tb_as1 in @tb1     select @tb_as1.@fd1'
    ),
]
print(f" {translate_examples=}")

t1 = vocab.encode('from tb1     in customer select tb1.name')
t1 = vocab.decode_values(t1)
print(f" {t1=}")
t1 = vocab.encode('from @tb_as1 in @tb1     select @tb_as1.@fd1')
t1 = vocab.decode_values(t1)
print(f" {t1=}")

translate_examples = [(vocab.encode(src), vocab.encode(tgt)) for (srg, tgt) in translate_examples]
print(f" {translate_examples=}")

t1 = [(vocab.decode_values(src), vocab.decode_values(tgt)) for (src, tgt) in translate_examples]
print(f" {t1=}")

copy_last_ckpt(LiTranslator)

model = start_train(LiTranslator,
                    {
                        'vocab_size': vocab.get_size(),
                        'padding_idx': vocab.get_value('<pad>')
                    },
                    ListDataset(translate_examples, vocab.get_value('<pad>')),
                    model_name='MyTrans',
                    device='cuda',
                    max_epochs=10)
# model = load_model(LiTranslator)
text = model.infer('from tb2 in p select tb2.name', vocab)
print(f"{text=}")
