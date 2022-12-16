import random

import torch
import torch.nn as nn

import logging
from ml.lit import BaseLightning


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, p=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension

        # 因為使用雙向的話會有 0 跟 1 所以這邊會需要cat起來
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        # encoder_states => 是最後的狀態, h1 h2 h3 放在hidden 裡面
        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embedding_size, hidden_size, num_layers, p=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size

        # hidden 應該要視為把encoder 的結果做 fc(h1 ,h2 h3)
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        #  encoder_states 應該為fc後的結果 = fc( final hidden state 3 or decoder_h1 h2 h3)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        # energy 應該為 s1 , s2 , s3，
        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)
        # attention score

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        # context_vector = attention weight * encoder_states

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, tgt_vocab_size, encoder, decoder):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder = encoder
        self.decoder = decoder
        self.logger = logging.getLogger("pytorch_lightning.core")
        self.logger.addHandler(logging.FileHandler("./output/train.log"))

    def log(self, msg):
        self.logger.log(20, msg)

    def forward(self, source, target, teacher_force_ratio=0.5):
        device = next(self.parameters()).device

        source = source.permute(1, 0)
        target = target.permute(1, 0)

        # print(source.shape)
        batch_size = source.shape[1]
        target_len = target.shape[1]

        outputs = torch.zeros(target_len, batch_size, self.tgt_vocab_size).to(device)

        encoder_states, hidden, cell = self.encoder(source)

        hidden = hidden.to(device)
        cell = cell.to(device)

        # First input will be <SOS> token
        x = target[0]
        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            self.log(f" {x=}")
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            self.log(f" {output=}")

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs


class LiSeq2Seq(BaseLightning):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 src_word_dim=128, tgt_word_dim=128,
                 hidden_size=512, num_layers=3, padding_idx=0,
                 enc_dropout=0., dec_dropout=0.):
        super().__init__()
        encoder_net = Encoder(src_vocab_size, src_word_dim, hidden_size, num_layers, enc_dropout)
        decoder_net = Decoder(tgt_vocab_size, tgt_word_dim, hidden_size, num_layers, dec_dropout)
        self.model = Seq2Seq(tgt_vocab_size, encoder_net, decoder_net)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)
        
    def training_step(self, batch, batch_idx):
        for param in self.model.parameters():
            param.requires_grad = True
        super().training_step(batch, batch_idx)

    def forward(self, batch):
        src, src_len, tgt, tgt_len = batch
        output = self.model(src, tgt)
        return output, tgt

    def _calculate_loss(self, batch, batch_idx):
        output, tgt = batch
        output = output[1:].reshape(-1, output.shape[2])
        target = tgt[1:].reshape(-1)
        loss = self.loss_fn(output, target)
        return loss

    def infer(self, vocab, text, max_length=None):
        device = self.get_device()
        max_length = vocab.get_size() if max_length is None else max_length
        text_to_indices = [vocab.get_value('<bos>')] + vocab.encode_to_tokens(text) + [vocab.get_value('<eos>')]
        sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

        # 取消梯度修正
        with torch.no_grad():
            outputs_encoder, hiddens, cells = self.model.encoder(sentence_tensor)

        # 先宣告 outputs ，然後裡面放一個開符號
        outputs = [vocab.get_value('<bos>')]

        for _ in range(max_length):
            previous_word = torch.LongTensor([outputs[-1]]).to(device)

            # seq2seq 的decoder 會把上一個 hidden_state 跟 cell 當作 input
            # 然後output 機率最大的
            with torch.no_grad():
                output, hiddens, cells = self.model.decoder(
                    previous_word, outputs_encoder, hiddens, cells
                )
                best_guess = output.argmax(1).item()

            # 機率最大的數值再把它放進output
            outputs.append(best_guess)

            # 如果是結束字元 eos 的話就中斷不然會一直預測下去
            if output.argmax(1).item() == vocab.get_value('<eos>'):
                break

        # 再把 vactor 轉成文字
        translated_sentence = vocab.decode(outputs)
        return translated_sentence


