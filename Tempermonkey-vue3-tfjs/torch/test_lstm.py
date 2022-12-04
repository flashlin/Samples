import random

import torch
from torch import nn
import torch.nn.functional as F
from common.io import info, get_file_by_lines_iter, info_error
from ml.lit import PositionalEncoding, load_model, copy_last_ckpt, start_train, BaseLightning
from ml.model_utils import detach_lstm_hidden_state
from ml.trans_linq2tsql import LinqToSqlVocab
from ml.translate_net import LiTranslator, TranslateListDataset, TranslateCsvDataset

SEQ_LEN = 5
SRC_WORD_DIM = 3
TGT_WORD_DIM = 3
MAX_SENTENCE_LEN = 1000
HIDDEN_SIZE = 7
NUM_LAYERS = 3
POS_DIM = SRC_WORD_DIM


class Encoder(nn.Module):
    def __init__(self,
                 pos_dim, max_sentence_len,
                 src_vocab_size, src_word_dim, src_padding_idx,
                 hidden_size=3,
                 num_layers=7,
                 dropout=0.1,
                 ):
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
        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_emb(tgt)
        outputs, hidden = self.lstm(tgt, hidden)
        pred = self.classify(outputs)
        # detach_lstm_hidden_state(hidden)
        return pred, hidden


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
        self.loss_fn = nn.CrossEntropyLoss()  # ignore_index=tgt_padding_idx)
        self.device = None

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        batch_size = y.size(0)
        target_len = y.size(1)
        device = next(self.parameters()).device
        self.device = device

        # tensor to store decoder outputs of each time step
        predicts = torch.zeros(y.shape).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        _, hidden = self.encoder(x)

        # first input to decoder is last coordinates of x
        decoder_input = y[:, -1]

        for i in range(target_len):
            # run decode for one time step
            outputs, hidden = self.decoder(decoder_input, hidden)
            # last_output = outputs[:, -1, :]   # 拿最後一個結果
            prob, pred = torch.max(outputs, dim=2, keepdim=False)
            # place predictions in a tensor holding predictions for each time step
            predicts[0, i] = pred

            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            decoder_input = y[:, i] if teacher_forcing else pred
            # decoder_input = pred

        return predicts

        #
        # outputs_save = torch.zeros(batch_size, tgt_seq_len, self.tgt_vocab_size)
        # _, hidden = self.encoder(src)
        #
        # tgt_i = tgt[:, 0]
        # for i in range(0, tgt_seq_len):
        #     output, hidden = self.decoder(tgt_i, hidden)
        #     tags = F.log_softmax(output, dim=1)
        #
        #     outputs_save[:, i, :] = tags
        #     print(f" decode {output.shape=} {tags.shape=} {top=}")
        #
        #     # outputs_save[:, i, :] = output
        #     # top = output.argmax(dim=0).argmax(dim=0)  # 拿到預測結果
        #     # print(f" {top=}")
        #     # top = max(top)
        #     probability = random.random()
        #     if probability > teach_rate:
        #         tgt_i = tgt[:, i]
        #     else:
        #         tgt_i = top
        #     # tgt_i = tgt[:, i] if probability > teach_rate else top
        # return outputs_save

    def calculate_loss(self, x_hat, y):
        # x_hat = x_hat.contiguous().view(-1, x_hat.size(-1))
        # y = y.contiguous().view(-1)
        y = y.type(torch.FloatTensor).to(self.device)
        return self.loss_fn(x_hat, y)


vocab = LinqToSqlVocab()

model = Seq2Seq(seq_len=SEQ_LEN,
                src_pos_dim=POS_DIM,
                max_sentence_len=MAX_SENTENCE_LEN,
                src_vocab_size=vocab.get_size(),
                src_word_dim=SRC_WORD_DIM,
                src_padding_idx=vocab.padding_idx,
                tgt_vocab_size=vocab.get_size(),
                tgt_word_dim=TGT_WORD_DIM,
                tgt_padding_idx=vocab.padding_idx,
                )

inp1_values = vocab.encode('from tb1 in p select tb1.name')
inp2_values = vocab.encode('@tb_as1 in @tb1 select @tb1.@fd1')

inp1_values = torch.tensor([inp1_values], dtype=torch.long)
inp2_values = torch.tensor([inp2_values], dtype=torch.long)
predictive = model(inp1_values, inp2_values)
print(f"{predictive=}")

loss = model.calculate_loss(predictive, inp2_values)
print(f" {loss=}")


class MySeq(BaseLightning):
    def __init__(self):
        super().__init__()
        self.model = Seq2Seq(seq_len=SEQ_LEN,
                             src_pos_dim=POS_DIM,
                             max_sentence_len=MAX_SENTENCE_LEN,
                             src_vocab_size=vocab.get_size(),
                             src_word_dim=SRC_WORD_DIM,
                             src_padding_idx=vocab.padding_idx,
                             tgt_vocab_size=vocab.get_size(),
                             tgt_word_dim=TGT_WORD_DIM,
                             tgt_padding_idx=vocab.padding_idx,
                             )

    def forward(self, batch):
        src, src_len, tgt, tgt_len = batch
        x_hat = self.model(src, tgt)
        return x_hat, tgt

    def _calculate_loss(self, data, mode="train"):
        (x_hat, y), batch = data
        loss = self.model.calculate_loss(x_hat, y)
        self.log("%s_loss" % mode, loss)
        return loss


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
